"""
DeepEval evaluation for tau-bench agent results.

Uses:
- ToolCorrectnessMetric: evaluates tool call correctness
- GEval: custom metrics for task completion and response quality

Usage:
    python eval_deepeval/run_eval.py results/your_results.json

Prerequisites:
    pip install -U deepeval
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import (
    ToolCorrectnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AmazonBedrockModel

from eval_common.extract_results import extract_eval_data, EvalDataPoint
from typing import List


def create_bedrock_model(model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
    """Create an AmazonBedrockModel for DeepEval."""
    return AmazonBedrockModel(
        model=model_id,
        region=os.getenv("AWS_REGION_NAME", "us-east-1"),
    )


def build_test_cases(eval_data: List[EvalDataPoint]) -> List[LLMTestCase]:
    """Convert eval data to DeepEval test cases."""
    test_cases = []

    for dp in eval_data:
        actual_tools = []
        for tc in dp.tool_calls:
            actual_tools.append(ToolCall(
                name=tc.name,
                input_parameters=tc.arguments if isinstance(tc.arguments, dict) else {},
                output=tc.result or "",
            ))

        expected_tools = []
        for action in dp.expected_actions:
            expected_tools.append(ToolCall(
                name=action.get("name", ""),
                input_parameters=action.get("kwargs", {}),
            ))

        expected_output = "; ".join(dp.expected_outputs) if dp.expected_outputs else "Task completed successfully"

        test_case = LLMTestCase(
            input=dp.task_instruction,
            actual_output=dp.agent_output,
            expected_output=expected_output,
            tools_called=actual_tools if actual_tools else None,
            expected_tools=expected_tools if expected_tools else None,
        )
        test_cases.append(test_case)

    return test_cases


def create_metrics(bedrock_model) -> list:
    """Create DeepEval metrics for evaluation."""
    metrics = []

    # 1. Tool correctness
    tool_correctness = ToolCorrectnessMetric(
        threshold=0.5,
        should_consider_ordering=False,
        should_exact_match=False,
        model=bedrock_model,
    )
    metrics.append(tool_correctness)

    # 2. Task completion (custom GEval)
    task_completion = GEval(
        name="TaskCompletion",
        evaluation_steps=[
            "Analyze the user's task instruction and determine what the agent was asked to do.",
            "Check if the agent's actual output addresses the user's request.",
            "Verify that the agent's response contains the expected information or confirms the expected actions were taken.",
            "Score 1.0 if the task is fully completed, 0.5 if partially completed, 0.0 if not completed.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=bedrock_model,
    )
    metrics.append(task_completion)

    # 3. Response quality (custom GEval)
    response_quality = GEval(
        name="ResponseQuality",
        evaluation_steps=[
            "Check if the response is professional and appropriate for customer service.",
            "Verify the response is clear and easy to understand.",
            "Check if the response directly addresses the user's needs without unnecessary information.",
            "Evaluate if the response follows proper customer service protocol (greeting, confirmation, etc.).",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=bedrock_model,
    )
    metrics.append(response_quality)

    return metrics


def run_evaluation(results_path: str, bedrock_model_id: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"):
    """Run DeepEval evaluation on tau-bench results."""
    print(f"Loading results from {results_path}...")
    eval_data = extract_eval_data(results_path)
    print(f"Loaded {len(eval_data)} evaluation data points.")

    avg_reward = sum(dp.reward for dp in eval_data) / len(eval_data)
    print(f"\n[Tau-bench baseline] Average reward: {avg_reward:.4f}")

    # Build test cases
    test_cases = build_test_cases(eval_data)
    print(f"Built {len(test_cases)} test cases.")

    # Create bedrock model and metrics
    bedrock_model = create_bedrock_model(bedrock_model_id)
    metrics = create_metrics(bedrock_model)
    metric_names = [m.name if hasattr(m, 'name') else type(m).__name__ for m in metrics]
    print(f"Using {len(metrics)} metrics: {metric_names}")

    # Run evaluation
    print("\nRunning DeepEval evaluation...")
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # State consistency (deterministic, computed outside DeepEval metrics)
    state_scores = []
    for dp in eval_data:
        state_scores.append(1.0 if dp.state_consistent else 0.0)
    avg_state = sum(state_scores) / len(state_scores) if state_scores else 0.0

    # Display summary
    print("\n" + "=" * 60)
    print("DeepEval Evaluation Summary")
    print("=" * 60)
    print(f"\n  State Consistency (deterministic): {avg_state:.4f}")

    eval_results = []
    for i, (dp, tc) in enumerate(zip(eval_data, test_cases)):
        print(f"\n  Task {dp.task_id}:")
        print(f"    tau-bench reward: {dp.reward}")
        print(f"    instruction: {dp.task_instruction[:100]}...")

        state_score = state_scores[i]
        print(f"    state_consistency: {state_score:.4f}")
        if dp.gt_data_hash:
            print(f"      gt_hash: {dp.gt_data_hash[:16]}... agent_hash: {dp.agent_data_hash[:16]}...")

        task_result = {
            "task_id": dp.task_id,
            "tau_bench_reward": dp.reward,
            "state_consistency": state_score,
            "gt_data_hash": dp.gt_data_hash,
            "agent_data_hash": dp.agent_data_hash,
            "metrics": {},
        }

        for metric in metrics:
            metric_name = metric.name if hasattr(metric, 'name') else type(metric).__name__
            try:
                metric.measure(tc)
                score = metric.score
                reason = getattr(metric, 'reason', '')
                print(f"    {metric_name}: {score:.4f}")
                if reason:
                    print(f"      Reason: {str(reason)[:200]}")
                task_result["metrics"][metric_name] = {
                    "score": score,
                    "reason": reason,
                    "passed": score >= metric.threshold if hasattr(metric, 'threshold') else None,
                }
            except Exception as e:
                print(f"    {metric_name}: ERROR - {e}")
                task_result["metrics"][metric_name] = {"error": str(e)}

        eval_results.append(task_result)

    # Save results
    output_path = results_path.replace(".json", "_deepeval_eval.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepEval evaluation on tau-bench results")
    parser.add_argument("results_path", help="Path to tau-bench results JSON file")
    parser.add_argument("--judge-model", default="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                        help="Bedrock model ID for LLM-based metrics")
    args = parser.parse_args()

    run_evaluation(args.results_path, args.judge_model)
