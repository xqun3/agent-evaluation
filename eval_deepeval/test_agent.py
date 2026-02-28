"""
DeepEval pytest-style tests for tau-bench agent evaluation.

Usage:
    uv run deepeval test run eval_deepeval/test_agent.py
    uv run deepeval test run eval_deepeval/test_agent.py --verbose-mode

Set RESULTS_PATH env var to specify the results file:
    RESULTS_PATH=results/your_results.json uv run deepeval test run eval_deepeval/test_agent.py

Prerequisites:
    uv add deepeval aiobotocore
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval import assert_test
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import (
    ToolCorrectnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AmazonBedrockModel

from eval_common.extract_results import extract_eval_data

# Configuration
RESULTS_PATH = os.getenv("RESULTS_PATH", "")
JUDGE_MODEL_ID = os.getenv("JUDGE_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")

# Create AmazonBedrockModel instance (lazy, created on first use)
_BEDROCK_MODEL = None


def get_bedrock_model():
    global _BEDROCK_MODEL
    if _BEDROCK_MODEL is None:
        _BEDROCK_MODEL = AmazonBedrockModel(
            model=JUDGE_MODEL_ID,
            region=os.getenv("AWS_REGION_NAME", "us-east-1"),
        )
    return _BEDROCK_MODEL


def _load_test_data():
    """Load eval data and build test cases."""
    if not RESULTS_PATH or not os.path.exists(RESULTS_PATH):
        # Find the most recent results file
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        if os.path.exists(results_dir):
            files = sorted(
                [f for f in os.listdir(results_dir) if f.endswith(".json") and "eval" not in f],
                key=lambda x: os.path.getmtime(os.path.join(results_dir, x)),
                reverse=True,
            )
            if files:
                path = os.path.join(results_dir, files[0])
            else:
                pytest.skip("No results files found in results/")
                return []
        else:
            pytest.skip("No results directory found")
            return []
    else:
        path = RESULTS_PATH

    eval_data = extract_eval_data(path)
    test_data = []

    for dp in eval_data:
        actual_tools = [
            ToolCall(
                name=tc.name,
                input_parameters=tc.arguments if isinstance(tc.arguments, dict) else {},
                output=tc.result or "",
            )
            for tc in dp.tool_calls
        ]

        expected_tools = [
            ToolCall(
                name=action.get("name", ""),
                input_parameters=action.get("kwargs", {}),
            )
            for action in dp.expected_actions
        ]

        # expected_output must not be None (DeepEval raises MissingTestCaseParamsError)
        expected_output = "; ".join(dp.expected_outputs) if dp.expected_outputs else "Task completed successfully"

        test_case = LLMTestCase(
            input=dp.task_instruction,
            actual_output=dp.agent_output,
            expected_output=expected_output,
            tools_called=actual_tools if actual_tools else None,
            expected_tools=expected_tools if expected_tools else None,
        )
        test_data.append((dp, test_case))

    return test_data


# Load data once at module level
_TEST_DATA = None


def get_test_data():
    global _TEST_DATA
    if _TEST_DATA is None:
        _TEST_DATA = _load_test_data()
    return _TEST_DATA


# ── Test: Tool Correctness ──


@pytest.mark.parametrize(
    "idx", range(10),  # Support up to 10 tasks
    ids=lambda i: f"task-{i}",
)
def test_tool_correctness(idx):
    """Test that the agent called the correct tools."""
    data = get_test_data()
    if idx >= len(data):
        pytest.skip(f"Only {len(data)} tasks available")

    dp, test_case = data[idx]

    if not test_case.expected_tools:
        pytest.skip("No expected tools for this task")

    metric = ToolCorrectnessMetric(
        threshold=0.5,
        should_consider_ordering=False,
        should_exact_match=False,
        model=get_bedrock_model(),
    )
    assert_test(test_case, [metric])


# ── Test: Task Completion ──


@pytest.mark.parametrize(
    "idx", range(10),
    ids=lambda i: f"task-{i}",
)
def test_task_completion(idx):
    """Test that the agent completed its task."""
    data = get_test_data()
    if idx >= len(data):
        pytest.skip(f"Only {len(data)} tasks available")

    dp, test_case = data[idx]

    metric = GEval(
        name="TaskCompletion",
        evaluation_steps=[
            "Analyze the user's task instruction and determine what the agent was asked to do.",
            "Check if the agent's actual output addresses the user's request.",
            "Verify that the agent's response contains the expected information.",
            "Score 1.0 if fully completed, 0.5 if partially, 0.0 if not completed.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
        model=get_bedrock_model(),
    )
    assert_test(test_case, [metric])


# ── Test: Response Quality ──


@pytest.mark.parametrize(
    "idx", range(10),
    ids=lambda i: f"task-{i}",
)
def test_response_quality(idx):
    """Test that the agent's response is high quality."""
    data = get_test_data()
    if idx >= len(data):
        pytest.skip(f"Only {len(data)} tasks available")

    dp, test_case = data[idx]

    metric = GEval(
        name="ResponseQuality",
        evaluation_steps=[
            "Check if the response is professional and appropriate for customer service.",
            "Verify the response is clear and easy to understand.",
            "Check if the response directly addresses the user's needs.",
            "Evaluate if proper customer service protocol is followed.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=get_bedrock_model(),
    )
    assert_test(test_case, [metric])


# ── Test: State Consistency ──


@pytest.mark.parametrize(
    "idx", range(10),
    ids=lambda i: f"task-{i}",
)
def test_state_consistency(idx):
    """Test that the agent's final database state matches the golden state."""
    data = get_test_data()
    if idx >= len(data):
        pytest.skip(f"Only {len(data)} tasks available")

    dp, test_case = data[idx]

    assert dp.state_consistent, (
        f"State mismatch: agent_hash={dp.agent_data_hash[:16]}... "
        f"gt_hash={dp.gt_data_hash[:16]}..."
    )
