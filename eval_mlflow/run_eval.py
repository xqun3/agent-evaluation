"""
MLflow GenAI evaluation for tau-bench agent results.

Uses MLflow 3.x scorers:
- Safety: checks for harmful content (LLM judge)
- Correctness: checks if agent output matches expected (LLM judge)
- Custom scorers: tool call accuracy, task completion, tool call efficiency

Usage:
    python eval_mlflow/run_eval.py results/your_results.json

Prerequisites:
    pip install 'mlflow[genai]>=3.3'
"""

import sys
import os
import json
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
from mlflow.genai.scorers import (
    Correctness,
    Safety,
)
from mlflow.genai import scorer
from mlflow.genai.scorers.base import Feedback

from eval_common.extract_results import extract_eval_data, EvalDataPoint
from typing import List


# ── Custom Scorers ────────────────────────────────────────────────


@scorer(name="tool_call_accuracy")
def tool_call_accuracy(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Check if the agent called the expected tools."""
    expected_tools = expectations.get("expected_tool_names", [])
    actual_tools = outputs.get("tool_names", [])

    if not expected_tools:
        return Feedback(score=1.0, rationale="No expected tool calls defined for this task.")

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    correct = expected_set & actual_set
    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:
        score = len(correct) / len(expected_set) if expected_set else 0.0
        rationale = f"Missing tools: {missing}. Extra tools: {extra}. Correct: {correct}."
    elif extra:
        score = 0.8
        rationale = f"All expected tools called, but extra tools used: {extra}."
    else:
        score = 1.0
        rationale = f"All expected tools correctly called: {correct}."

    return Feedback(score=score, rationale=rationale)


@scorer(name="task_completion")
def task_completion(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Check if expected outputs appear in agent response (tau-bench style)."""
    expected_outputs = expectations.get("expected_outputs", [])
    agent_output = outputs.get("agent_output", "").lower().replace(",", "")

    if not expected_outputs:
        return Feedback(score=1.0, rationale="No expected outputs defined.")

    found = []
    missing = []
    for exp in expected_outputs:
        if exp.lower() in agent_output:
            found.append(exp)
        else:
            missing.append(exp)

    score = len(found) / len(expected_outputs) if expected_outputs else 1.0
    rationale = f"Found {len(found)}/{len(expected_outputs)} expected outputs."
    if missing:
        rationale += f" Missing: {missing}"

    return Feedback(score=score, rationale=rationale)


@scorer(name="tool_call_efficiency")
def tool_call_efficiency(inputs: dict, outputs: dict) -> Feedback:
    """Check for redundant tool calls."""
    tool_names = outputs.get("tool_names", [])
    if not tool_names:
        return Feedback(score=1.0, rationale="No tool calls made.")

    counts = Counter(tool_names)
    repeated = {name: count for name, count in counts.items()
                if count > 1 and name not in ("think",)}

    if not repeated:
        return Feedback(score=1.0, rationale=f"No redundant tool calls. Total: {len(tool_names)}.")

    score = max(0.0, 1.0 - 0.2 * len(repeated))
    return Feedback(
        score=score,
        rationale=f"Repeated tool calls: {repeated}. Total calls: {len(tool_names)}."
    )


@scorer(name="state_consistency")
def state_consistency(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    """Check if agent's final database state matches the golden state.

    Uses pre-computed hashes from extract_results replay logic.
    """
    gt_hash = expectations.get("gt_data_hash", "")
    agent_hash = expectations.get("agent_data_hash", "")
    consistent = expectations.get("state_consistent", False)

    if not gt_hash or not agent_hash:
        return Feedback(score=0.0, rationale="State hashes not available for this task.")

    if consistent:
        return Feedback(score=1.0, rationale=f"Database state matches golden state. Hash: {gt_hash[:16]}...")
    else:
        return Feedback(
            score=0.0,
            rationale=f"Database state mismatch. Agent hash: {agent_hash[:16]}... Golden hash: {gt_hash[:16]}...",
        )


# ── Data Preparation ─────────────────────────────────────────────


def prepare_mlflow_data(eval_data: List[EvalDataPoint]) -> list:
    """Convert extracted eval data to MLflow evaluation format."""
    records = []
    for dp in eval_data:
        user_messages = [m["content"] for m in dp.conversation if m["role"] == "user"]
        query = user_messages[0] if user_messages else dp.task_instruction

        expected_tool_names = [a.get("name", "") for a in dp.expected_actions]
        actual_tool_names = [tc.name for tc in dp.tool_calls]

        records.append({
            "inputs": {
                "question": query,
                "task_instruction": dp.task_instruction,
            },
            "outputs": {
                "answer": dp.agent_output,
                "agent_output": dp.agent_output,
                "tool_names": actual_tool_names,
            },
            "expectations": {
                "expected_response": "; ".join(dp.expected_outputs) if dp.expected_outputs else dp.task_instruction,
                "expected_outputs": dp.expected_outputs,
                "expected_tool_names": expected_tool_names,
                "gt_data_hash": dp.gt_data_hash,
                "agent_data_hash": dp.agent_data_hash,
                "state_consistent": dp.state_consistent,
            }
        })

    return records


# ── Main ─────────────────────────────────────────────────────────


def run_evaluation(results_path: str, experiment_name: str = "tau-bench-eval",
                    judge_model: str = "bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
    """Run MLflow evaluation on tau-bench results."""
    print(f"Loading results from {results_path}...")
    eval_data = extract_eval_data(results_path)
    print(f"Loaded {len(eval_data)} evaluation data points.")

    mlflow_data = prepare_mlflow_data(eval_data)
    print(f"Prepared {len(mlflow_data)} records for MLflow evaluation.")

    avg_reward = sum(dp.reward for dp in eval_data) / len(eval_data)
    print(f"\n[Tau-bench baseline] Average reward: {avg_reward:.4f}")

    mlflow.set_experiment(experiment_name)

    # Phase 1: Custom scorers via evaluate()
    custom_scorers = [tool_call_accuracy, task_completion, tool_call_efficiency, state_consistency]
    print("\nRunning MLflow evaluation (custom scorers)...")
    with mlflow.start_run(run_name="tau-bench-mlflow-eval"):
        results = mlflow.genai.evaluate(
            data=mlflow_data,
            scorers=custom_scorers,
        )

    # Phase 2: LLM-based scorers called directly per record
    # (mlflow.genai.evaluate with LLM scorers hangs in threaded mode with bedrock)
    print("\nRunning LLM-based scorers directly...")
    safety_scorer = Safety(model=judge_model)
    correctness_scorer = Correctness(model=judge_model)

    llm_results = {}
    for i, record in enumerate(mlflow_data):
        output_text = record["outputs"]["answer"]
        llm_results[i] = {}

        # Safety
        try:
            safety_fb = safety_scorer(outputs=output_text)
            score_val = safety_fb.feedback.value if hasattr(safety_fb, 'feedback') else str(safety_fb)
            llm_results[i]["safety"] = {
                "score": score_val,
                "rationale": safety_fb.rationale if hasattr(safety_fb, 'rationale') else "",
            }
        except Exception as e:
            llm_results[i]["safety"] = {"score": "error", "rationale": str(e)}

        # Correctness
        try:
            corr_fb = correctness_scorer(
                inputs=record["inputs"],
                outputs=output_text,
                expectations={"expected_response": record["expectations"]["expected_response"]},
            )
            score_val = corr_fb.feedback.value if hasattr(corr_fb, 'feedback') else str(corr_fb)
            llm_results[i]["correctness"] = {
                "score": score_val,
                "rationale": corr_fb.rationale if hasattr(corr_fb, 'rationale') else "",
            }
        except Exception as e:
            llm_results[i]["correctness"] = {"score": "error", "rationale": str(e)}

    # Display results
    print("\n" + "=" * 60)
    print("MLflow Evaluation Results")
    print("=" * 60)

    print("\n── Custom Scorer Metrics (via evaluate) ──")
    if results.metrics:
        for metric_name, value in sorted(results.metrics.items()):
            print(f"  {metric_name}: {value}")
    else:
        print("  (Metrics logged to MLflow traces - view in MLflow UI)")

    # Show per-task from eval_results
    table_key = "eval_results" if "eval_results" in results.tables else next(iter(results.tables), None)
    if table_key:
        df = results.tables[table_key]
        print(f"\n── Per-Task Custom Scorer Results (table: {table_key}) ──")
        for col in df.columns:
            if "score" in col.lower() or "rationale" in col.lower():
                for idx, val in df[col].items():
                    print(f"  [{idx}] {col}: {val}")

    print("\n── LLM-based Scorer Results ──")
    for i, record in enumerate(mlflow_data):
        instruction = record["inputs"].get("task_instruction", "")[:80]
        print(f"\n  Task {i}: {instruction}...")
        for scorer_name, result in llm_results[i].items():
            print(f"    {scorer_name}: {result['score']}")
            if result['rationale']:
                print(f"      Rationale: {str(result['rationale'])[:200]}")

    print(f"\nMLflow UI: run 'mlflow ui' and open http://localhost:5000")

    # Save combined results
    output_path = results_path.replace(".json", "_mlflow_eval.json")
    save_data = {"llm_scorers": llm_results, "tau_bench_reward": avg_reward}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLflow evaluation on tau-bench results")
    parser.add_argument("results_path", help="Path to tau-bench results JSON file")
    parser.add_argument("--experiment", default="tau-bench-eval", help="MLflow experiment name")
    parser.add_argument("--judge-model", default="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        help="Model for LLM-based scorers (mlflow format: provider:/model-name)")
    args = parser.parse_args()

    run_evaluation(args.results_path, args.experiment, args.judge_model)
