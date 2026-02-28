"""
Generic state consistency evaluation module for Strands Agent projects.

Provides MockAgent + replay infrastructure so that tools can use standard
agent.state.get/set paths during both normal execution and evaluation replay,
eliminating the need for per-tool hacks.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Dict, List, Tuple, Union


# ── Hashable conversion (mirrors tau_bench.envs.base) ────────────

ToHashable = Union[
    str, int, float, Dict[str, Any], List[Any],
]


def _to_hashable(item):
    """Convert nested data to a hashable representation."""
    if isinstance(item, dict):
        return tuple((key, _to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, (list, tuple)):
        return tuple(_to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(_to_hashable(element) for element in item))
    else:
        return item


def _consistent_hash(value) -> str:
    """SHA256 hash."""
    return sha256(str(value).encode("utf-8")).hexdigest()


def get_data_hash(data: Dict[str, Any]) -> str:
    """Compute deterministic hash for a data dict."""
    return _consistent_hash(_to_hashable(data))


# ── MockAgent: allows tools to use standard agent.state.get/set ──

class _MockState:
    """Supports both get/set (Strands 0.1.x) and dict-style access (Strands 1.x)."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value


class MockAgent:
    """Provides agent.state.get/set interface so tools run unmodified."""

    def __init__(self, state_data: Any, state_key: str = "datas"):
        self.state = _MockState({state_key: state_data})
        self.messages: list = []


# ── Config & replay ──────────────────────────────────────────────

@dataclass
class StateEvaluatorConfig:
    """Configuration for state consistency evaluation.

    Args:
        state_factory: Callable that returns a fresh copy of the initial state (e.g. load_data).
        tools: Mapping from tool name to tool function with Strands signature
               ``(tool_use, agent, **kwargs) -> ToolResult``.
        state_key: Key under which the state dict lives in agent.state (default "datas").
        terminate_tools: Tool names to skip during replay (e.g. transfer_to_human_agents).
    """
    state_factory: Callable[[], Dict[str, Any]]
    tools: Dict[str, Callable]
    state_key: str = "datas"
    terminate_tools: List[str] = field(default_factory=list)


def replay_actions(
    actions: List[Dict[str, Any]],
    config: StateEvaluatorConfig,
) -> Tuple[Dict[str, Any], str]:
    """Replay a list of actions on a fresh state copy via MockAgent.

    Each action is ``{"name": "tool_name", "kwargs": {...}}``.

    Returns:
        (final_state_dict, hash_of_final_state)
    """
    state = config.state_factory()
    mock = MockAgent(state, config.state_key)

    for action in actions:
        name = action.get("name", "")
        if name in config.terminate_tools:
            continue
        tool_func = config.tools.get(name)
        if tool_func is None:
            continue
        tool_use = {
            "toolUseId": "replay",
            "input": action.get("kwargs", {}),
        }
        try:
            tool_func(tool_use, mock)
        except Exception:
            pass  # tool errors don't stop replay

    final_state = mock.state.get(config.state_key)
    return final_state, get_data_hash(final_state)


@dataclass
class StateConsistencyResult:
    """Result of comparing agent vs golden state after replay."""
    agent_hash: str
    golden_hash: str
    consistent: bool


def evaluate_state_consistency(
    agent_actions: List[Dict[str, Any]],
    golden_actions: List[Dict[str, Any]],
    config: StateEvaluatorConfig,
) -> StateConsistencyResult:
    """Replay both action lists on fresh state copies and compare hashes."""
    _, agent_hash = replay_actions(agent_actions, config)
    _, golden_hash = replay_actions(golden_actions, config)
    return StateConsistencyResult(
        agent_hash=agent_hash,
        golden_hash=golden_hash,
        consistent=(agent_hash == golden_hash),
    )
