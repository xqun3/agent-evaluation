# eval_common — 通用评估公共模块

本目录提供 Agent 评估的公共基础设施，包括**状态一致性评估**和**结果提取标准化**。三个评估框架（MLflow、agentevals、DeepEval）均基于本模块工作。

## 文件说明

| 文件 | 职责 |
|------|------|
| `state_eval.py` | 通用状态一致性评估：MockAgent、replay、hash 对比 |
| `extract_results.py` | 从 JSON 结果文件提取标准化评估数据（`EvalDataPoint`） |
| `__init__.py` | 统一导出 |

## 架构概览

```
results/*.json (Agent 运行产生的原始结果)
       ↓
extract_results.py → 提取 EvalDataPoint 列表
       ↓                  ↓
       ↓            state_eval.py → 重放工具调用 → hash 对比 → state_consistent
       ↓
┌──────────────┬──────────────┬──────────────┐
│ eval_mlflow/ │ eval_langsmith/ │ eval_deepeval/ │
│  读取 dp.*   │  读取 dp.*      │  读取 dp.*     │
└──────────────┴──────────────┴──────────────┘
```

---

## 1. state_eval.py — 通用状态一致性评估

### 核心思路

Agent 执行任务时会通过工具调用修改系统状态（如数据库）。评估时，我们在**全新的数据库副本**上分别重放 agent 的工具调用和 golden 期望动作，然后对比两者的最终状态 hash 是否一致。

关键设计：工具在正常运行时通过 `agent.state.get("datas")` / `agent.state.set("datas", datas)` 读写状态。重放时，`state_eval.py` 构造一个 `MockAgent`，提供相同的 `state.get/set` 接口，**工具代码无需任何修改**即可在重放模式下运行。

### API

#### StateEvaluatorConfig

评估配置，描述如何加载数据和调用工具：

```python
from eval_common.state_eval import StateEvaluatorConfig

config = StateEvaluatorConfig(
    state_factory=load_data,                      # Callable，返回全新数据库副本
    tools={"cancel_pending_order": cancel_func},  # Dict[str, Callable]，工具名 → 工具函数
    state_key="datas",                            # agent.state 中数据的 key（默认 "datas"）
    terminate_tools=["transfer_to_human_agents"],  # 重放时跳过的工具（可选）
)
```

**参数说明：**

- `state_factory`：每次调用返回一份**独立的**初始状态，如 `data.load_data`。
- `tools`：工具函数必须是 Strands 签名 `(tool_use: dict, agent, **kwargs) -> ToolResult`，其中 `tool_use["input"]` 包含参数。
- `state_key`：工具通过 `agent.state.get(state_key)` 访问数据的 key。
- `terminate_tools`：终止类工具（如转人工），重放时跳过。

#### replay_actions

在全新数据库副本上重放一组 actions：

```python
from eval_common.state_eval import replay_actions

actions = [
    {"name": "get_order_details", "kwargs": {"order_id": "#W2378156"}},
    {"name": "cancel_pending_order", "kwargs": {"order_id": "#W2378156", "reason": "no longer needed"}},
]

final_state, state_hash = replay_actions(actions, config)
# final_state: 重放后的数据库状态 dict
# state_hash: 该状态的 SHA256 hash
```

#### evaluate_state_consistency

对比 agent 和 golden 重放后的状态：

```python
from eval_common.state_eval import evaluate_state_consistency

result = evaluate_state_consistency(agent_actions, golden_actions, config)

print(result.consistent)    # True / False
print(result.agent_hash)    # agent 重放后的 hash
print(result.golden_hash)   # golden 重放后的 hash
```

#### get_data_hash

对任意数据 dict 计算确定性 hash：

```python
from eval_common.state_eval import get_data_hash

h = get_data_hash({"orders": {...}, "users": {...}})
```

#### MockAgent

供工具重放时使用，提供 `agent.state.get/set` 接口：

```python
from eval_common.state_eval import MockAgent

mock = MockAgent(state_data=load_data(), state_key="datas")
tool_func(tool_use, mock)               # 工具正常调用 mock.state.get("datas")
print(mock.state.get("datas"))           # 查看工具修改后的状态
```

### 完整示例

```python
from eval_common.state_eval import StateEvaluatorConfig, evaluate_state_consistency
from data import load_data
from tools import TOOL_MAP

# 1. 创建配置
config = StateEvaluatorConfig(
    state_factory=load_data,
    tools=TOOL_MAP,
    state_key="datas",
    terminate_tools=["transfer_to_human_agents"],
)

# 2. 定义 agent 实际执行的工具调用和 golden 期望动作
agent_actions = [
    {"name": "find_user_id_by_email", "kwargs": {"email": "john@example.com"}},
    {"name": "get_order_details", "kwargs": {"order_id": "#W2378156"}},
    {"name": "cancel_pending_order", "kwargs": {"order_id": "#W2378156", "reason": "no longer needed"}},
]

golden_actions = [
    {"name": "cancel_pending_order", "kwargs": {"order_id": "#W2378156", "reason": "no longer needed"}},
]

# 3. 评估
result = evaluate_state_consistency(agent_actions, golden_actions, config)
print(f"状态一致: {result.consistent}")
# 即使 agent 多调用了 find_user 和 get_order（只读操作），
# 只要最终数据库状态相同，就判定为一致。
```

### 接入你自己的 Strands Agent 项目

只需确保：

1. 你的工具函数是 Strands 签名：`def my_tool(tool: ToolUse, agent: Agent, **kwargs) -> ToolResult`
2. 工具通过 `agent.state.get("your_key")` / `agent.state.set("your_key", value)` 读写状态
3. 你有一个 `state_factory` 函数能返回全新的初始状态

```python
from eval_common.state_eval import StateEvaluatorConfig, evaluate_state_consistency

config = StateEvaluatorConfig(
    state_factory=my_load_data,          # 你的数据加载函数
    tools={"my_tool": my_tool_func},     # 你的工具映射
    state_key="my_state_key",            # 你的 state key
)
result = evaluate_state_consistency(agent_actions, golden_actions, config)
```

---

## 2. extract_results.py — 结果提取与标准化

### 核心数据结构

#### EvalDataPoint

从 JSON 结果文件提取的标准化评估数据，三个评估框架共用：

```python
@dataclass
class EvalDataPoint:
    task_id: int
    trial: int
    reward: float                           # 原始 reward（0.0 或 1.0）

    conversation: List[Dict[str, str]]      # [{"role": "user/assistant", "content": "..."}]
    tool_calls: List[ToolCallRecord]        # agent 的工具调用记录
    agent_output: str                       # agent 最后一条回复

    task_instruction: str                   # 任务指令
    expected_outputs: List[str]             # 期望输出关键词
    expected_actions: List[Dict[str, Any]]  # 期望动作 [{"name": ..., "kwargs": ...}]

    gt_data_hash: str                       # golden 状态 hash
    agent_data_hash: str                    # agent 重放后 hash
    state_consistent: bool                  # 两者是否相等
```

#### ToolCallRecord

单次工具调用记录：

```python
@dataclass
class ToolCallRecord:
    name: str                               # 工具名
    arguments: Dict[str, Any]               # 调用参数
    result: Optional[str]                   # 返回结果
```

### API

#### extract_eval_data

主函数，加载 JSON 结果文件 → 提取标准化数据 → 计算状态一致性：

```python
from eval_common.extract_results import extract_eval_data

# 方式 1：使用自定义 config（通用，任何 Strands 项目）
from eval_common.state_eval import StateEvaluatorConfig
config = StateEvaluatorConfig(state_factory=load_data, tools=TOOL_MAP, state_key="datas")
eval_data = extract_eval_data("results/my_results.json", config=config)

# 方式 2：不传 config（tau-bench 向后兼容，自动检测环境类型）
eval_data = extract_eval_data("results/my_results.json")
```

返回 `List[EvalDataPoint]`，其中 `gt_data_hash`、`agent_data_hash`、`state_consistent` 已预计算好。

#### load_results

只加载原始 JSON，不做提取：

```python
from eval_common.extract_results import load_results
raw = load_results("results/my_results.json")  # List[Dict]
```

#### to_openai_trajectory

将工具调用转为 OpenAI message 格式（agentevals 需要）：

```python
from eval_common.extract_results import to_openai_trajectory
messages = to_openai_trajectory(dp.tool_calls)
# [{"role": "assistant", "tool_calls": [...]}, {"role": "tool", "content": "..."}]
```

#### replay_and_compute_hashes

tau-bench 兼容接口，封装了环境检测和重放逻辑：

```python
from eval_common.extract_results import replay_and_compute_hashes
gt_hash, agent_hash, consistent = replay_and_compute_hashes(
    tool_calls, expected_actions, env_type="retail"
)
```

#### create_tau_bench_config

为 tau-bench 环境创建 `StateEvaluatorConfig`（将 tau-bench 的 `Tool.invoke` 适配为 Strands 工具签名）：

```python
from eval_common.extract_results import create_tau_bench_config
config = create_tau_bench_config("retail")  # 或 "airline"
```

---

## 3. 各评估框架如何使用 eval_common

三个评估框架的调用模式一致：调用 `extract_eval_data()` 获取 `EvalDataPoint` 列表，然后读取预计算好的字段。

### agentevals (`eval_langsmith/run_eval.py`)

```python
from eval_common.extract_results import extract_eval_data

eval_data = extract_eval_data(results_path)

# 状态一致性评估 — 直接读 dp.state_consistent
for dp in eval_data:
    score = 1.0 if dp.state_consistent else 0.0
```

### MLflow (`eval_mlflow/run_eval.py`)

```python
from eval_common.extract_results import extract_eval_data

eval_data = extract_eval_data(results_path)

# 将状态 hash 放入 expectations，传给 @scorer
record = {
    "expectations": {
        "gt_data_hash": dp.gt_data_hash,
        "agent_data_hash": dp.agent_data_hash,
        "state_consistent": dp.state_consistent,
    }
}
```

### DeepEval (`eval_deepeval/run_eval.py`)

```python
from eval_common.extract_results import extract_eval_data

eval_data = extract_eval_data(results_path)

# 独立计算状态一致性分数
state_scores = [1.0 if dp.state_consistent else 0.0 for dp in eval_data]
```

---

## 4. 运行验证

```bash
# 验证 state_eval 模块
uv run python -c "
from eval_common.state_eval import StateEvaluatorConfig, evaluate_state_consistency
from data import load_data
from tools import TOOL_MAP

config = StateEvaluatorConfig(state_factory=load_data, tools=TOOL_MAP, state_key='datas')
result = evaluate_state_consistency(
    [{'name': 'get_order_details', 'kwargs': {'order_id': '#W2378156'}}],
    [{'name': 'get_order_details', 'kwargs': {'order_id': '#W2378156'}}],
    config
)
print(f'consistent={result.consistent}')  # True
"

# 验证 extract_results（需要有结果文件）
uv run python eval_common/extract_results.py results/<your_results>.json

# 运行三个评估框架
uv run python eval_langsmith/run_eval.py results/<your_results>.json
uv run python eval_mlflow/run_eval.py results/<your_results>.json
uv run python eval_deepeval/run_eval.py results/<your_results>.json
```

---

## 5. JSON 结果文件格式

`extract_eval_data` 期望的 JSON 格式（与 tau-bench 输出一致）：

```json
[
  {
    "task_id": 0,
    "trial": 0,
    "reward": 1.0,
    "traj": [
      {"role": "user", "content": [{"text": "Hi!"}]},
      {"role": "assistant", "content": [
        {"text": "Hello!"},
        {"toolUse": {"toolUseId": "abc", "name": "get_order_details", "input": {"order_id": "#W001"}}}
      ]},
      {"role": "user", "content": [
        {"toolResult": {"toolUseId": "abc", "content": [{"text": "{...}"}]}}
      ]}
    ],
    "info": {
      "task": {
        "instruction": "用户要求取消订单 #W001",
        "outputs": ["#W001"],
        "actions": [
          {"name": "cancel_pending_order", "kwargs": {"order_id": "#W001", "reason": "no longer needed"}}
        ]
      }
    }
  }
]
```
