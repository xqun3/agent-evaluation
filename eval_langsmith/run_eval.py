"""
LangSmith / agentevals evaluation for tau-bench agent results.

Uses:
- agentevals: trajectory matching (strict, unordered, superset, subset)
- agentevals: trajectory LLM-as-judge
- openevals: LLM-as-judge correctness

Usage:
    python eval_langsmith/run_eval.py results/your_results.json

Prerequisites:
    pip install openevals agentevals
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentevals.trajectory.match import create_trajectory_match_evaluator
from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from eval_common.extract_results import (
    extract_eval_data,
    to_openai_trajectory,
    EvalDataPoint,
)
from typing import List, Dict, Any


def build_reference_trajectory(dp: EvalDataPoint) -> List[Dict]:
    """Build a reference trajectory from expected actions."""
    messages = []
    for action in dp.expected_actions:
        name = action.get("name", "")
        kwargs = action.get("kwargs", {})
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": name,
                    "arguments": json.dumps(kwargs) if isinstance(kwargs, dict) else str(kwargs),
                }
            }]
        })
        # Add a placeholder tool response
        messages.append({
            "role": "tool",
            "content": "(expected result)",
        })
    return messages


def run_trajectory_match_eval(eval_data: List[EvalDataPoint]) -> Dict[str, Any]:
    """Run all 4 trajectory match modes."""
    modes = ["strict", "unordered", "superset", "subset"]
    results = {}

    for mode in modes:
        evaluator = create_trajectory_match_evaluator(
            trajectory_match_mode=mode,
            tool_args_match_mode="ignore",  # Focus on tool selection, not exact args
        )
        scores = []
        for dp in eval_data:
            actual_traj = to_openai_trajectory(dp.tool_calls)
            reference_traj = build_reference_trajectory(dp)

            if not reference_traj:
                scores.append(1.0)
                continue

            try:
                result = evaluator(
                    outputs=actual_traj,
                    reference_outputs=reference_traj,
                )
                score = result.get("score", 0) if isinstance(result, dict) else getattr(result, "score", 0)
                scores.append(float(score) if score is not None else 0.0)
            except Exception as e:
                print(f"  Warning: trajectory_{mode}_match failed for task {dp.task_id}: {e}")
                scores.append(0.0)

        avg = sum(scores) / len(scores) if scores else 0.0
        results[f"trajectory_{mode}_match"] = {
            "average": avg,
            "scores": scores,
        }

    return results


def run_trajectory_llm_judge(eval_data: List[EvalDataPoint], model: str = "bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0") -> Dict[str, Any]:
    """Run LLM-as-judge on trajectories."""
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
        model=model,
        feedback_key="trajectory_accuracy",
    )

    scores = []
    reasonings = []
    for dp in eval_data:
        actual_traj = to_openai_trajectory(dp.tool_calls)
        reference_traj = build_reference_trajectory(dp)

        try:
            result = evaluator(
                inputs={"question": dp.task_instruction},
                outputs=actual_traj,
                reference_outputs=reference_traj,
            )
            score = result.get("score", 0) if isinstance(result, dict) else getattr(result, "score", 0)
            reasoning = result.get("reasoning", "") if isinstance(result, dict) else getattr(result, "reasoning", "")
            scores.append(float(score) if score is not None else 0.0)
            reasonings.append(str(reasoning))
        except Exception as e:
            print(f"  Warning: trajectory LLM judge failed for task {dp.task_id}: {e}")
            scores.append(0.0)
            reasonings.append(str(e))

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "trajectory_llm_judge": {
            "average": avg,
            "scores": scores,
            "reasonings": reasonings,
        }
    }


def run_correctness_eval(eval_data: List[EvalDataPoint], model: str = "bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0") -> Dict[str, Any]:
    """Run LLM-as-judge correctness on final outputs."""
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        model=model,
        feedback_key="correctness",
    )

    scores = []
    reasonings = []
    for dp in eval_data:
        reference = "; ".join(dp.expected_outputs) if dp.expected_outputs else "Task completed successfully"
        try:
            result = evaluator(
                inputs={"question": dp.task_instruction},
                outputs={"answer": dp.agent_output},
                reference_outputs={"answer": reference},
            )
            score = result.get("score", 0) if isinstance(result, dict) else getattr(result, "score", 0)
            reasoning = result.get("reasoning", "") if isinstance(result, dict) else getattr(result, "reasoning", "")
            scores.append(float(score) if score is not None else 0.0)
            reasonings.append(str(reasoning))
        except Exception as e:
            print(f"  Warning: correctness eval failed for task {dp.task_id}: {e}")
            scores.append(0.0)
            reasonings.append(str(e))

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "correctness": {
            "average": avg,
            "scores": scores,
            "reasonings": reasonings,
        }
    }


def run_state_consistency_eval(eval_data: List[EvalDataPoint]) -> Dict[str, Any]:
    """Deterministic state consistency check — no LLM needed.

    Compares the database hash after replaying agent tool calls vs golden actions.
    """
    scores = []
    for dp in eval_data:
        scores.append(1.0 if dp.state_consistent else 0.0)

    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "state_consistency": {
            "average": avg,
            "scores": scores,
        }
    }


def run_evaluation(results_path: str, judge_model: str = "bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
    """Run full LangSmith/agentevals evaluation."""
    print(f"Loading results from {results_path}...")
    eval_data = extract_eval_data(results_path)
    print(f"Loaded {len(eval_data)} evaluation data points.")

    avg_reward = sum(dp.reward for dp in eval_data) / len(eval_data)
    print(f"\n[Tau-bench baseline] Average reward: {avg_reward:.4f}")

    all_results = {}

    # 1. Trajectory match (deterministic, no LLM needed)
    print("\n── Running trajectory match evaluation ──")
    match_results = run_trajectory_match_eval(eval_data)
    all_results.update(match_results)
    for mode, res in match_results.items():
        print(f"  {mode}: {res['average']:.4f}")

    # 2. Trajectory LLM-as-judge
    print("\n── Running trajectory LLM-as-judge ──")
    llm_results = run_trajectory_llm_judge(eval_data, model=judge_model)
    all_results.update(llm_results)
    print(f"  trajectory_llm_judge: {llm_results['trajectory_llm_judge']['average']:.4f}")

    # 3. Correctness LLM-as-judge
    print("\n── Running correctness evaluation ──")
    corr_results = run_correctness_eval(eval_data, model=judge_model)
    all_results.update(corr_results)
    print(f"  correctness: {corr_results['correctness']['average']:.4f}")

    # 4. State consistency (deterministic, no LLM needed)
    print("\n── Running state consistency evaluation ──")
    state_results = run_state_consistency_eval(eval_data)
    all_results.update(state_results)
    print(f"  state_consistency: {state_results['state_consistency']['average']:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("LangSmith/agentevals Evaluation Summary")
    print("=" * 60)
    for metric_name, metric_data in all_results.items():
        print(f"  {metric_name}: {metric_data['average']:.4f}")

    # Per-task details
    print("\n── Per-Task Details ──")
    for i, dp in enumerate(eval_data):
        print(f"\n  Task {dp.task_id}:")
        print(f"    tau-bench reward: {dp.reward}")
        for metric_name, metric_data in all_results.items():
            score = metric_data["scores"][i] if i < len(metric_data["scores"]) else "N/A"
            print(f"    {metric_name}: {score}")
            if "reasonings" in metric_data and i < len(metric_data["reasonings"]):
                reasoning = metric_data["reasonings"][i]
                if reasoning:
                    print(f"      Reasoning: {str(reasoning)[:200]}")

    # Save results
    output_path = results_path.replace(".json", "_langsmith_eval.json")
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {
            "average": v["average"],
            "scores": v["scores"],
        }
        if "reasonings" in v:
            serializable[k]["reasonings"] = v["reasonings"]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSmith/agentevals evaluation")
    parser.add_argument("results_path", help="Path to tau-bench results JSON file")
    parser.add_argument("--judge-model", default="bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        help="Model for LLM-as-judge evaluations (langchain format: provider:model_id)")
    args = parser.parse_args()

    run_evaluation(args.results_path, args.judge_model)
