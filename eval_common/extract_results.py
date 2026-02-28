"""
Extract standardized evaluation data from tau-bench JSON results.

Provides a common data format for all three evaluation frameworks:
- Conversation history
- Tool call trajectories
- Agent final output
- Expected outputs/actions
"""

import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

# Add parent dir so we can import tau_bench types
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_common.state_eval import (
    get_data_hash,
    StateEvaluatorConfig,
    replay_actions,
    evaluate_state_consistency,
)


@dataclass
class ToolCallRecord:
    """A single tool call made by the agent."""
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None


@dataclass
class EvalDataPoint:
    """Standardized evaluation data for a single task run."""
    task_id: int
    trial: int
    reward: float  # 0.0 or 1.0 from tau-bench

    # Conversation
    conversation: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]

    # Tool calls made by agent
    tool_calls: List[ToolCallRecord]

    # Agent's final output (last assistant message)
    agent_output: str

    # Expected (from task definition)
    task_instruction: str
    expected_outputs: List[str]
    expected_actions: List[Dict[str, Any]]

    # Raw info dict for framework-specific use
    raw_info: Dict[str, Any] = field(default_factory=dict)

    # State consistency fields
    gt_data_hash: str = ""        # golden state hash (from replaying expected actions)
    agent_data_hash: str = ""     # agent tool calls replayed on fresh DB
    state_consistent: bool = False  # agent_data_hash == gt_data_hash


# ── State consistency: replay & hash ─────────────────────────────
# get_data_hash, replay_actions, evaluate_state_consistency are imported from state_eval


def _detect_env(info: Dict[str, Any]) -> str:
    """Detect environment type (retail or airline) from task info."""
    task = info.get("task", {})
    actions = task.get("actions", [])
    action_names = {a.get("name", "") if isinstance(a, dict) else "" for a in actions}

    airline_tools = {"book_reservation", "cancel_reservation", "search_direct_flight",
                     "get_reservation_details", "update_reservation_flights"}
    if action_names & airline_tools:
        return "airline"
    return "retail"


def create_tau_bench_config(env_type: str) -> StateEvaluatorConfig:
    """Create a StateEvaluatorConfig for tau-bench environments (backward-compatible).

    Wraps tau-bench Tool.invoke(data=..., **kwargs) into Strands tool signature
    (tool_use, agent, **kw) so the generic replay infrastructure can drive them.
    """
    if env_type == "airline":
        from tau_bench.envs.airline.data import load_data
        from tau_bench.envs.airline.tools import ALL_TOOLS
    else:
        from tau_bench.envs.retail.data import load_data
        from tau_bench.envs.retail.tools import ALL_TOOLS

    def _adapt(tool_cls):
        def wrapper(tool_use, agent, **kw):
            data = agent.state.get("datas")
            result = tool_cls.invoke(data=data, **tool_use["input"])
            agent.state.set("datas", data)
            return result
        return wrapper

    tools = {
        t.get_info()["function"]["name"]: _adapt(t) for t in ALL_TOOLS
    }
    return StateEvaluatorConfig(state_factory=load_data, tools=tools)


def replay_and_compute_hashes(
    tool_calls: List,
    expected_actions: List[Dict[str, Any]],
    env_type: str = "retail",
) -> Tuple[str, str, bool]:
    """Replay agent and golden actions on fresh DB copies, return hashes.

    Returns:
        (gt_data_hash, agent_data_hash, state_consistent)
    """
    config = create_tau_bench_config(env_type)

    # Convert ToolCallRecords to dicts if needed
    actions_for_replay = []
    for action in tool_calls:
        if isinstance(action, dict):
            actions_for_replay.append(action)
        else:
            actions_for_replay.append({
                "name": action.name,
                "kwargs": action.arguments if isinstance(action.arguments, dict) else {},
            })

    _, agent_hash = replay_actions(actions_for_replay, config)
    _, gt_hash = replay_actions(expected_actions, config)

    return gt_hash, agent_hash, (gt_hash == agent_hash)


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load raw tau-bench results from JSON file."""
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_conversation(traj: List[Dict]) -> List[Dict[str, str]]:
    """Extract simplified conversation from trajectory messages."""
    conversation = []
    for msg in traj:
        role = msg.get("role", "")
        content_parts = msg.get("content", [])

        if isinstance(content_parts, str):
            conversation.append({"role": role, "content": content_parts})
            continue

        text_parts = []
        for part in content_parts:
            if isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)

        if text_parts:
            conversation.append({"role": role, "content": "\n".join(text_parts)})

    return conversation


def _extract_tool_calls(traj: List[Dict]) -> List[ToolCallRecord]:
    """Extract tool calls from trajectory messages."""
    tool_calls = []
    tool_results = {}

    # First pass: collect tool results
    for msg in traj:
        content_parts = msg.get("content", [])
        if not isinstance(content_parts, list):
            continue
        for part in content_parts:
            if isinstance(part, dict) and "toolResult" in part:
                tr = part["toolResult"]
                tool_use_id = tr.get("toolUseId", "")
                result_content = tr.get("content", [])
                result_text = ""
                for rc in result_content:
                    if isinstance(rc, dict) and "text" in rc:
                        result_text += rc["text"]
                tool_results[tool_use_id] = result_text

    # Second pass: collect tool calls
    for msg in traj:
        content_parts = msg.get("content", [])
        if not isinstance(content_parts, list):
            continue
        for part in content_parts:
            if isinstance(part, dict) and "toolUse" in part:
                tu = part["toolUse"]
                tool_use_id = tu.get("toolUseId", "")
                tool_calls.append(ToolCallRecord(
                    name=tu.get("name", ""),
                    arguments=tu.get("input", {}),
                    result=tool_results.get(tool_use_id),
                ))

    return tool_calls


def _extract_expected(info: Dict[str, Any]) -> tuple:
    """Extract expected outputs and actions from task info."""
    task_info = info.get("task", {})
    expected_outputs = task_info.get("outputs", [])
    expected_actions = []
    for action in task_info.get("actions", []):
        if isinstance(action, dict):
            expected_actions.append(action)
        elif hasattr(action, "model_dump"):
            expected_actions.append(action.model_dump())
    instruction = task_info.get("instruction", "")
    return instruction, expected_outputs, expected_actions


def extract_eval_data(
    results_path: str,
    config: Optional[StateEvaluatorConfig] = None,
) -> List[EvalDataPoint]:
    """
    Load results and extract standardized evaluation data.

    Args:
        results_path: Path to the results JSON file.
        config: Optional StateEvaluatorConfig for state consistency replay.
                If provided, uses this config for all tasks (any Strands project).
                If None, falls back to auto-detecting tau-bench environment type.

    Returns:
        List of EvalDataPoint objects ready for evaluation
    """
    raw_results = load_results(results_path)
    eval_data = []

    for result in raw_results:
        traj = result.get("traj", [])
        info = result.get("info", {})

        conversation = _extract_conversation(traj)
        tool_calls = _extract_tool_calls(traj)

        # Get last assistant message as final output
        agent_output = ""
        for msg in reversed(conversation):
            if msg["role"] == "assistant":
                agent_output = msg["content"]
                break

        instruction, expected_outputs, expected_actions = _extract_expected(info)

        # Compute state consistency via replay
        try:
            # Convert ToolCallRecords to dicts for replay
            agent_actions_for_replay = [
                {"name": tc.name, "kwargs": tc.arguments if isinstance(tc.arguments, dict) else {}}
                for tc in tool_calls
            ]
            if config is not None:
                # Generic path: use caller-provided config
                result_obj = evaluate_state_consistency(
                    agent_actions_for_replay, expected_actions, config
                )
                gt_hash = result_obj.golden_hash
                agent_hash = result_obj.agent_hash
                consistent = result_obj.consistent
            else:
                # tau-bench fallback: auto-detect environment
                gt_hash, agent_hash, consistent = replay_and_compute_hashes(
                    agent_actions_for_replay, expected_actions, _detect_env(info)
                )
        except Exception as e:
            print(f"  Warning: state consistency computation failed for task {result.get('task_id', -1)}: {e}")
            gt_hash, agent_hash, consistent = "", "", False

        eval_data.append(EvalDataPoint(
            task_id=result.get("task_id", -1),
            trial=result.get("trial", 0),
            reward=result.get("reward", 0.0),
            conversation=conversation,
            tool_calls=tool_calls,
            agent_output=agent_output,
            task_instruction=instruction,
            expected_outputs=expected_outputs,
            expected_actions=expected_actions,
            raw_info=info,
            gt_data_hash=gt_hash,
            agent_data_hash=agent_hash,
            state_consistent=consistent,
        ))

    return eval_data


def to_openai_trajectory(tool_calls: List[ToolCallRecord]) -> List[Dict]:
    """
    Convert tool calls to OpenAI message format (for agentevals).

    Returns list of messages in the format:
    [
        {"role": "assistant", "tool_calls": [{"function": {"name": ..., "arguments": ...}}]},
        {"role": "tool", "content": "..."},
        ...
    ]
    """
    messages = []
    for tc in tool_calls:
        messages.append({
            "role": "assistant",
            "tool_calls": [{
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments),
                }
            }]
        })
        if tc.result is not None:
            messages.append({
                "role": "tool",
                "content": tc.result,
            })
    return messages


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python extract_results.py <results.json>")
        sys.exit(1)

    data = extract_eval_data(sys.argv[1])
    for dp in data:
        print(f"\n=== Task {dp.task_id} (trial {dp.trial}) ===")
        print(f"Reward: {dp.reward}")
        print(f"Instruction: {dp.task_instruction[:100]}...")
        print(f"Tool calls: {[tc.name for tc in dp.tool_calls]}")
        print(f"Agent output: {dp.agent_output[:200]}...")
        print(f"Expected outputs: {dp.expected_outputs}")
        print(f"Expected actions: {[a.get('name', '') for a in dp.expected_actions]}")
        print(f"State consistent: {dp.state_consistent} (gt={dp.gt_data_hash[:16]}... agent={dp.agent_data_hash[:16]}...)")
