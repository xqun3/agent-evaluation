# Agent 评估框架测试报告

> 本文档记录了使用 MLflow、LangSmith/agentevals、DeepEval 三个框架对 tau-bench 零售客服 Agent 进行评估的完整过程，包含每个框架的输入输出示例、差异分析和使用指南。

---

## 目录

1. [测试环境与流程](#1-测试环境与流程)
2. [Agent 运行结果（评估输入）](#2-agent-运行结果评估输入)
3. [公共数据提取层](#3-公共数据提取层)
4. [LangSmith/agentevals 评估](#4-langsmithagentevalsopenevals-评估)
5. [MLflow 评估](#5-mlflow-评估)
6. [DeepEval 评估](#6-deepeval-评估)
7. [三框架对比分析](#7-三框架对比分析)
8. [踩坑记录与注意事项](#8-踩坑记录与注意事项)

---

## 1. 测试环境与流程

### 1.1 环境

| 项目 | 值 |
|------|-----|
| Python | 3.13.5 |
| 包管理 | uv 0.8.11 |
| Agent 框架 | Strands Agents |
| 评估基准 | tau-bench (retail) |
| Agent 模型 | `us.anthropic.claude-3-5-haiku-20241022-v1:0` (Bedrock) |
| 评估 Judge 模型 | `global.anthropic.claude-sonnet-4-5-20250929-v1:0` (Bedrock) |

### 1.2 整体流程

```
                        ┌─────────────────────┐
                        │  python main.py     │
                        │  --task-ids 0       │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  results/*.json     │
                        │  (对话轨迹 + 奖励)    │
                        └──────────┬──────────┘
                                   │
                        ┌──────────┴──────────┐
                        │  eval_common/       │
                        │  extract_results.py │
                        │  (标准化提取)         │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
    │  eval_langsmith │  │  eval_mlflow    │  │  eval_deepeval  │
    │  (agentevals +  │  │  (MLflow 3.10   │  │  (DeepEval +    │
    │   openevals)    │  │   GenAI)        │  │   Bedrock)      │
    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### 1.3 运行命令

```bash
# Step 1: 安装依赖
uv sync

# Step 2: 运行 Agent（单条任务）
uv run python main.py \
  --agent-strategy tool-calling \
  --env retail \
  --model us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --model-provider bedrock \
  --user-model us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --user-model-provider bedrock \
  --user-strategy llm \
  --task-ids 0 \
  --max-concurrency 1

# Step 3: 运行三个评估
uv run python eval_langsmith/run_eval.py results/<file>.json
uv run python eval_mlflow/run_eval.py results/<file>.json
uv run python eval_deepeval/run_eval.py results/<file>.json
```

---

## 2. Agent 运行结果（评估输入）

### 2.1 测试任务

**Task 0** — 商品换货场景：

> 你是 Yusuf Rossi，邮编 19122。你收到了订单 #W2378156，想把机械键盘换成 clicky 轴的，把智能温控器换成兼容 Google Home 的（而不是 Apple HomeKit）。如果没有 clicky+RGB+全尺寸的键盘，你愿意选无背光版本。你注重细节，希望一次性解决所有问题。

### 2.2 期望工具调用序列（Ground Truth）

```
1. find_user_id_by_name_zip(first_name="Yusuf", last_name="Rossi", zip="19122")
2. get_order_details(order_id="#W2378156")
3. get_product_details(product_id="1656367028")    # 键盘
4. get_product_details(product_id="4896585277")    # 温控器
5. exchange_delivered_order_items(
     order_id="#W2378156",
     item_ids=["1151293680", "4983901480"],
     new_item_ids=["7706410293", "7747408585"],
     payment_method_id="credit_card_9513926"
   )
```

### 2.3 Agent 实际工具调用

```
1. get_order_details(order_id="#W2378156")          ← 跳过了身份验证
2. get_product_details(product_id="1656367028")
3. get_product_details(product_id="4896585277")
4. exchange_delivered_order_items(...)               ← 参数完全正确
```

**差异**：Agent 跳过了 `find_user_id_by_name_zip`（身份验证步骤），但换货操作本身完全正确。

### 2.4 tau-bench 评估结果

```
reward: 1.0  ✅
```

tau-bench 通过数据库状态 hash 对比判断任务是否完成，Agent 虽然跳过了身份验证，但数据库最终状态正确，所以得分 1.0。

---

## 3. 公共数据提取层

### 3.1 `eval_common/extract_results.py`

从 tau-bench 的 JSON 结果中提取标准化数据，供三个框架共用。

**核心数据结构**：

```python
@dataclass
class EvalDataPoint:
    task_id: int
    trial: int
    reward: float                           # tau-bench 原始奖励

    conversation: List[Dict[str, str]]      # [{"role": "user/assistant", "content": "..."}]
    tool_calls: List[ToolCallRecord]        # Agent 实际调用的工具
    agent_output: str                       # Agent 最后一条回复

    task_instruction: str                   # 任务指令
    expected_outputs: List[str]             # 期望输出文本
    expected_actions: List[Dict[str, Any]]  # 期望工具调用
```

**提取示例输出**：

```
task_id: 0
reward: 1.0
task_instruction: You are Yusuf Rossi in 19122. You received your order #W2378156...
expected_outputs: []
expected_actions:
  - find_user_id_by_name_zip({"first_name":"Yusuf","last_name":"Rossi","zip":"19122"})
  - get_order_details({"order_id":"#W2378156"})
  - get_product_details({"product_id":"1656367028"})
  - get_product_details({"product_id":"4896585277"})
  - exchange_delivered_order_items({...})
tool_calls (实际):
  - get_order_details({"order_id":"#W2378156"})
  - get_product_details({"product_id":"1656367028"})
  - get_product_details({"product_id":"4896585277"})
  - exchange_delivered_order_items({...})
```

### 3.2 OpenAI 轨迹格式转换

`to_openai_trajectory()` 将工具调用转换为 OpenAI message 格式，供 agentevals 使用：

```json
[
  {
    "role": "assistant",
    "tool_calls": [{
      "function": {
        "name": "get_order_details",
        "arguments": "{\"order_id\": \"#W2378156\"}"
      }
    }]
  },
  {
    "role": "tool",
    "content": "{\"order_id\": \"#W2378156\", \"user_id\": \"yusuf_rossi_9620\", ...}"
  },
  ...
]
```

---

## 4. LangSmith/agentevals/openevals 评估

### 4.1 安装

```bash
uv add openevals agentevals langchain-aws
```

### 4.2 评估指标

| 指标 | 类型 | 是否需要 LLM | 说明 |
|------|------|------------|------|
| trajectory_strict_match | 确定性 | 否 | 工具调用序列严格匹配（顺序+参数） |
| trajectory_unordered_match | 确定性 | 否 | 工具调用集合匹配（不考虑顺序） |
| trajectory_superset_match | 确定性 | 否 | 实际调用是期望调用的超集 |
| trajectory_subset_match | 确定性 | 否 | 实际调用是期望调用的子集 |
| trajectory_llm_judge | LLM Judge | 是 | LLM 评估轨迹整体准确性 |
| correctness | LLM Judge | 是 | LLM 评估回答正确性 |

### 4.3 核心代码

**轨迹匹配（无需 LLM）**：

```python
from agentevals.trajectory.match import create_trajectory_match_evaluator

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="subset",       # strict / unordered / superset / subset
    tool_args_match_mode="ignore",        # 只匹配工具名，忽略参数
)

result = evaluator(
    outputs=actual_trajectory,            # OpenAI message 格式
    reference_outputs=expected_trajectory,
)
# result.score = True/False 或 0.0-1.0
```

**LLM-as-Judge 轨迹评估**：

```python
from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)

evaluator = create_trajectory_llm_as_judge(
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    model="bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    feedback_key="trajectory_accuracy",
)

result = evaluator(
    inputs={"question": task_instruction},
    outputs=actual_trajectory,
    reference_outputs=expected_trajectory,
)
# result["score"] = True/False, result["reasoning"] = "..."
```

**LLM-as-Judge 正确性（openevals）**：

```python
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    feedback_key="correctness",
)

result = evaluator(
    inputs={"question": task_instruction},
    outputs={"answer": agent_output},
    reference_outputs={"answer": expected_answer},
)
```

> **注意**：agentevals/openevals 基于 LangChain，模型格式为 `provider:model_id`，Bedrock 需要用 `bedrock_converse:` 前缀，并安装 `langchain-aws`。

### 4.4 完整输入输出

**输入**：

```python
# 实际轨迹（Agent 调用了 4 个工具）
actual_trajectory = [
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_order_details", "arguments": "{\"order_id\": \"#W2378156\"}"}}]},
    {"role": "tool", "content": "{\"order_id\": \"#W2378156\", ...}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_product_details", "arguments": "{\"product_id\": \"1656367028\"}"}}]},
    {"role": "tool", "content": "{\"name\": \"Mechanical Keyboard\", ...}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_product_details", "arguments": "{\"product_id\": \"4896585277\"}"}}]},
    {"role": "tool", "content": "{\"name\": \"Smart Thermostat\", ...}"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "exchange_delivered_order_items", "arguments": "{...}"}}]},
    {"role": "tool", "content": "{\"order_id\": \"#W2378156\", \"status\": \"exchange requested\", ...}"}
]

# 期望轨迹（Ground Truth 有 5 个工具调用）
reference_trajectory = [
    {"role": "assistant", "tool_calls": [{"function": {"name": "find_user_id_by_name_zip", "arguments": "{...}"}}]},
    {"role": "tool", "content": "(expected result)"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_order_details", "arguments": "{...}"}}]},
    {"role": "tool", "content": "(expected result)"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_product_details", "arguments": "{...}"}}]},
    {"role": "tool", "content": "(expected result)"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "get_product_details", "arguments": "{...}"}}]},
    {"role": "tool", "content": "(expected result)"},
    {"role": "assistant", "tool_calls": [{"function": {"name": "exchange_delivered_order_items", "arguments": "{...}"}}]},
    {"role": "tool", "content": "(expected result)"}
]
```

**输出**：

```json
{
  "trajectory_strict_match":   { "average": 0.0, "scores": [0.0] },
  "trajectory_unordered_match":{ "average": 0.0, "scores": [0.0] },
  "trajectory_superset_match": { "average": 0.0, "scores": [0.0] },
  "trajectory_subset_match":   { "average": 1.0, "scores": [1.0] },
  "trajectory_llm_judge":      { "average": 1.0, "scores": [1.0] },
  "correctness":               { "average": 0.0, "scores": [0.0] }
}
```

### 4.5 结果解读

| 指标 | 分数 | 解读 |
|------|------|------|
| strict_match = 0.0 | ❌ | 实际调用了 4 个工具 ≠ 期望的 5 个工具（顺序+集合都不同） |
| unordered_match = 0.0 | ❌ | 即使不看顺序，工具集合也不完全相同（缺少 `find_user_id_by_name_zip`） |
| superset_match = 0.0 | ❌ | 实际调用不是期望的超集（实际少了一个工具） |
| **subset_match = 1.0** | ✅ | 实际调用的所有工具都在期望集合内（是子集） |
| **llm_judge = 1.0** | ✅ | LLM 认为整体轨迹是正确的，跳过身份验证不影响任务完成 |
| correctness = 0.0 | ❌ | 此任务没有 expected_outputs，用了 fallback 文本，导致匹配失败 |

---

## 5. MLflow 评估

### 5.1 安装

```bash
uv add 'mlflow[genai]>=3.3'
```

### 5.2 评估指标

| 指标 | 类型 | 说明 |
|------|------|------|
| Correctness | 内置 LLM Judge | 检查输出是否支持期望事实 |
| Safety | 内置 LLM Judge | 检查是否有有害内容 |
| tool_call_accuracy | 自定义 | 工具集合匹配度 |
| task_completion | 自定义 | 期望输出是否出现在回复中 |
| tool_call_efficiency | 自定义 | 是否有冗余工具调用 |

### 5.3 核心代码

**自定义 Scorer**：

```python
from mlflow.genai import scorer
from mlflow.genai.scorers.base import Feedback

@scorer(name="tool_call_accuracy")
def tool_call_accuracy(inputs: dict, outputs: dict, expectations: dict) -> Feedback:
    expected_tools = expectations.get("expected_tool_names", [])
    actual_tools = outputs.get("tool_names", [])
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)
    correct = expected_set & actual_set
    missing = expected_set - actual_set
    score = len(correct) / len(expected_set) if expected_set else 1.0
    return Feedback(
        score=score,
        rationale=f"Matched {correct}, missing {missing}"
    )
```

**内置 LLM Scorer**：

```python
from mlflow.genai.scorers import Correctness, Safety

safety = Safety(model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0")
result = safety(outputs="I can help with your order.")
# result.feedback.value = "yes"
# result.rationale = "The text is a standard customer service interaction..."

correctness = Correctness(model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0")
result = correctness(
    inputs={"question": task_instruction},
    outputs=agent_output,
    expectations={"expected_response": expected_text},
)
```

> **注意**：MLflow 的模型 URI 格式为 `provider:/model-name`（注意是 `:/` 不是 `/`），如 `bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0`。

**evaluate() 用法**：

```python
import mlflow

data = [{
    "inputs": {"question": "...", "task_instruction": "..."},
    "outputs": {"answer": "...", "tool_names": [...]},
    "expectations": {"expected_response": "...", "expected_tool_names": [...]},
}]

mlflow.set_experiment("tau-bench-eval")
with mlflow.start_run(run_name="eval-run"):
    results = mlflow.genai.evaluate(
        data=data,
        scorers=[tool_call_accuracy, task_completion, tool_call_efficiency],
    )
```

> **已知问题**：MLflow 3.10 的 `evaluate()` 在使用内置 LLM Scorer（如 Safety、Correctness）+ Bedrock 时会卡住（线程池与 boto3 session 的兼容问题）。解决方案是将 LLM Scorer 单独直接调用，自定义 Scorer 通过 `evaluate()` 执行。

### 5.4 完整输入输出

**输入**（evaluate() 的 data 格式）：

```python
[{
    "inputs": {
        "question": "Hi, I need help with exchanging some items from my order.",
        "task_instruction": "You are Yusuf Rossi in 19122. You received your order #W2378156..."
    },
    "outputs": {
        "answer": "Yes, that's correct! Let me confirm the details for you:\n\n1. You will receive an email with instructions...",
        "agent_output": "Yes, that's correct! ...",
        "tool_names": ["get_order_details", "get_product_details", "get_product_details", "exchange_delivered_order_items"]
    },
    "expectations": {
        "expected_response": "You are Yusuf Rossi in 19122...",
        "expected_outputs": [],
        "expected_tool_names": ["find_user_id_by_name_zip", "get_order_details", "get_product_details", "get_product_details", "exchange_delivered_order_items"]
    }
}]
```

**输出**：

```json
{
  "llm_scorers": {
    "0": {
      "safety": {
        "score": "yes",
        "rationale": "This text appears to be a customer service response confirming details about a product return and refund. It discusses returning a mechanical keyboard and smart thermostat, processing a refund to a credit card, and offering further assistance. The content is purely transactional and administrative in nature. There is no hate speech, harassment, incitement of violence, or promotion of illegal or harmful acts."
      },
      "correctness": {
        "score": "yes",
        "rationale": "The claim states that the user is Yusuf Rossi in 19122 who received order #W2378156 and wishes to exchange a mechanical keyboard for one with clicky switches and a smart thermostat for one compatible with Google Home... All elements of the claim are directly supported by the information in the document."
      }
    }
  },
  "tau_bench_reward": 1.0
}
```

### 5.5 结果解读

| 指标 | 分数 | 解读 |
|------|------|------|
| **Safety = "yes"** | ✅ | 内容安全，标准客服回复 |
| **Correctness = "yes"** | ✅ | LLM 综合判断回答正确，覆盖了用户需求 |
| 自定义 scorers | 记录到 MLflow traces | 需通过 `mlflow ui` 查看详细结果 |

---

## 6. DeepEval 评估

### 6.1 安装

```bash
uv add deepeval aiobotocore
```

### 6.2 评估指标

| 指标 | 类型 | 说明 |
|------|------|------|
| ToolCorrectnessMetric | 内置 | 工具调用正确性（名称+参数匹配） |
| TaskCompletion (GEval) | 自定义 LLM Judge | 任务是否完成 |
| ResponseQuality (GEval) | 自定义 LLM Judge | 回复质量评分 |

### 6.3 核心代码

**构建 TestCase**：

```python
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="You are Yusuf Rossi in 19122. You received your order...",
    actual_output="Yes, that's correct! Let me confirm the details...",
    expected_output="Task completed successfully",
    tools_called=[
        ToolCall(name="get_order_details",
                 input_parameters={"order_id": "#W2378156"},
                 output='{"order_id": "#W2378156", ...}'),
        ToolCall(name="get_product_details",
                 input_parameters={"product_id": "1656367028"},
                 output='{"name": "Mechanical Keyboard", ...}'),
        ToolCall(name="get_product_details",
                 input_parameters={"product_id": "4896585277"},
                 output='{"name": "Smart Thermostat", ...}'),
        ToolCall(name="exchange_delivered_order_items",
                 input_parameters={"order_id": "#W2378156", "item_ids": [...], ...},
                 output='{"order_id": "#W2378156", "status": "exchange requested", ...}'),
    ],
    expected_tools=[
        ToolCall(name="find_user_id_by_name_zip",
                 input_parameters={"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"}),
        ToolCall(name="get_order_details",
                 input_parameters={"order_id": "#W2378156"}),
        ToolCall(name="get_product_details",
                 input_parameters={"product_id": "1656367028"}),
        ToolCall(name="get_product_details",
                 input_parameters={"product_id": "4896585277"}),
        ToolCall(name="exchange_delivered_order_items",
                 input_parameters={"order_id": "#W2378156", ...}),
    ],
)
```

**定义指标**：

```python
from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AmazonBedrockModel

bedrock_model = AmazonBedrockModel(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region="us-east-1",
)

metrics = [
    ToolCorrectnessMetric(
        threshold=0.5,
        should_consider_ordering=False,
        should_exact_match=False,
        model=bedrock_model,
    ),
    GEval(
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
        model=bedrock_model,
    ),
    GEval(
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
        model=bedrock_model,
    ),
]
```

**运行评估**：

```python
from deepeval import evaluate

results = evaluate(test_cases=[test_case], metrics=metrics)
```

**pytest 风格（另一种方式）**：

```python
# eval_deepeval/test_agent.py
from deepeval import assert_test

def test_tool_correctness():
    assert_test(test_case, [ToolCorrectnessMetric(threshold=0.5)])
```

```bash
RESULTS_PATH=results/<file>.json deepeval test run eval_deepeval/test_agent.py
```

### 6.4 完整输出

```
Metrics Summary

  - ✅ Tool Correctness (score: 0.8, threshold: 0.5, strict: False)
    reason: Incomplete tool usage: missing tools [find_user_id_by_name_zip, ...];
            expected ['find_user_id_by_name_zip', 'get_order_details',
            'get_product_details', 'get_product_details',
            'exchange_delivered_order_items'],
            called ['get_order_details', 'get_product_details',
            'get_product_details', 'exchange_delivered_order_items']

  - ❌ TaskCompletion [GEval] (score: 0.3, threshold: 0.5)
    reason: The actual output appears to be a confirmation message in the middle
            of a conversation, but it does not demonstrate that the agent actually
            completed the exchange task. The user requested to exchange a mechanical
            keyboard for one with clicky switches and a smart thermostat for a
            Google Home compatible version. The actual output only confirms return
            instructions and refund details without showing that the agent
            identified suitable replacement products.

  - ❌ ResponseQuality [GEval] (score: 0.2, threshold: 0.5)
    reason: The response lacks a proper greeting and fails to address the
            customer's primary request. The customer wants to exchange items
            and has specific requirements, but the response only confirms
            return instructions and a refund without discussing the new items.

Overall: Pass Rate: 0.0% | Passed: 0 | Failed: 1
```

保存的 JSON 结果：

```json
[{
  "task_id": 0,
  "tau_bench_reward": 1.0,
  "metrics": {
    "ToolCorrectnessMetric": {
      "score": 0.8,
      "reason": "Incomplete tool usage: missing tools [...]; expected 5, called 4.",
      "passed": true
    },
    "TaskCompletion": {
      "score": 0.3,
      "reason": "The actual output appears to be a confirmation message...",
      "passed": false
    },
    "ResponseQuality": {
      "score": 0.2,
      "reason": "The response lacks a proper greeting and fails to address...",
      "passed": false
    }
  }
}]
```

### 6.5 结果解读

| 指标 | 分数 | Pass | 解读 |
|------|------|------|------|
| **ToolCorrectness = 0.8** | ✅ | 4/5 工具正确，缺少 `find_user_id_by_name_zip` |
| TaskCompletion = 0.3 | ❌ | GEval 只看到最后一条 agent 消息（确认信息），**缺少完整对话上下文**，误判为未完成 |
| ResponseQuality = 0.2 | ❌ | 同上，最后一条消息是确认而非完整交互，被判定质量低 |

> **重要发现**：DeepEval 的 GEval 指标 `actual_output` 只包含 Agent 的最后一条消息。在多轮对话场景中，最后一条消息往往是确认信息而非完整回答，导致 LLM Judge 严重低估任务完成度。如果要更准确的评估，需要将完整对话历史拼接作为 `actual_output`。

---

## 6.6 状态一致性评估（State Consistency）

### 状态评估 vs 轨迹评估

tau-bench 的核心评估思想是**最终系统状态一致性**：不关心 agent 走了什么路径，只看最终数据库状态是否与 golden state 一致。这与传统的轨迹评估形成互补：

| 维度 | 轨迹评估 | 状态评估 |
|------|---------|---------|
| 评估对象 | 工具调用序列、输出文本 | 数据库最终状态 hash |
| 容错性 | 严格匹配路径，灵活度低 | 允许多条正确路径，只看结果 |
| 典型场景 | agent 跳过身份验证但完成了任务 → 轨迹评估扣分 | 只要数据库状态正确 → 满分 |
| LLM 依赖 | 部分指标需要 LLM | 纯确定性，无需 LLM |

### 实现原理

核心逻辑全部在公共层 `eval_common/extract_results.py`，三个框架只是读取预计算好的结果。

**第一层：公共层的 replay + hash**（`eval_common/extract_results.py`）

```python
def replay_and_compute_hashes(tool_calls, expected_actions, env_type="retail"):
    load_data, tools_map = _load_env_resources(env_type)

    # 1. 新数据库副本 → 重放 agent 工具调用 → agent_hash
    agent_data = load_data()
    _replay_actions_on_data(agent_data, tool_calls, tools_map)
    agent_hash = get_data_hash(agent_data)

    # 2. 新数据库副本 → 重放 golden 期望动作 → gt_hash
    gt_data = load_data()
    _replay_actions_on_data(gt_data, expected_actions, tools_map)
    gt_hash = get_data_hash(gt_data)

    # 3. 两个 hash 一致 = 状态一致
    return gt_hash, agent_hash, (gt_hash == agent_hash)
```

重放通过逐个调用 tau-bench 的原始工具函数实现（`Tool.invoke(data=data, **kwargs)`），工具会直接修改 `data` dict，最后对整个 dict 做 SHA256 hash：

```python
def _replay_actions_on_data(data, actions, tools_map):
    for action in actions:
        name = action.get("name", "")
        kwargs = action.get("kwargs", {})
        if name in tools_map:
            try:
                tools_map[name].invoke(data=data, **kwargs)
            except Exception:
                pass  # 工具报错不中断重放
    return data

def get_data_hash(data):
    return sha256(str(_to_hashable(data)).encode("utf-8")).hexdigest()
```

在 `extract_eval_data()` 提取阶段，每个数据点就已经算好了三个字段：

```python
dp.gt_data_hash      # golden 状态 hash
dp.agent_data_hash   # agent 重放后 hash
dp.state_consistent  # bool: 两者是否相等
```

**第二层：三个框架各自的包装方式**

都是读 `dp.state_consistent` 这个预计算好的 bool，纯确定性，无需 LLM。

**agentevals**（`eval_langsmith/run_eval.py`）— 独立函数：

```python
def run_state_consistency_eval(eval_data):
    scores = []
    for dp in eval_data:
        scores.append(1.0 if dp.state_consistent else 0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    return {"state_consistency": {"average": avg, "scores": scores}}
```

**MLflow**（`eval_mlflow/run_eval.py`）— `@scorer` 装饰器，通过 `evaluate()` 执行：

```python
@scorer(name="state_consistency")
def state_consistency(inputs, outputs, expectations):
    consistent = expectations.get("state_consistent", False)
    gt_hash = expectations.get("gt_data_hash", "")
    agent_hash = expectations.get("agent_data_hash", "")
    if consistent:
        return Feedback(score=1.0, rationale=f"State matches. Hash: {gt_hash[:16]}...")
    else:
        return Feedback(score=0.0, rationale=f"State mismatch.")
```

hash 值通过 `prepare_mlflow_data()` 放入 `expectations` dict 传入。

**DeepEval**（`eval_deepeval/run_eval.py`）— 在 `evaluate()` 之外独立计算：

```python
state_scores = []
for dp in eval_data:
    state_scores.append(1.0 if dp.state_consistent else 0.0)
avg_state = sum(state_scores) / len(state_scores) if state_scores else 0.0
```

结果写入 summary 输出和保存的 JSON 中。

> **总结**：公共层做重活（加载新 DB → 重放工具调用 → SHA256 对比），三个框架只是用各自的方式包装 `dp.state_consistent` 这个 bool 值输出分数。

### 各框架输出示例

**agentevals/openevals**：
```
── Running state consistency evaluation ──
  state_consistency: 1.0000
```

**MLflow**：
```
state_consistency/score: 1.0
state_consistency/rationale: Database state matches golden state. Hash: 5ed21c65cf9e8d0c...
```

**DeepEval**：
```
  State Consistency (deterministic): 1.0000
  Task 0:
    state_consistency: 1.0000
      gt_hash: 5ed21c65cf9e8d0c... agent_hash: 5ed21c65cf9e8d0c...
```

---

## 7. 三框架对比分析

### 7.1 同一任务评估结果对比

| 评估维度 | tau-bench | agentevals | MLflow | DeepEval |
|----------|-----------|------------|--------|----------|
| 整体任务是否完成 | ✅ 1.0 | subset_match ✅ 1.0 | correctness ✅ yes | TaskCompletion ❌ 0.3 |
| 工具调用正确性 | ✅ (数据库 hash 匹配) | strict ❌ 0.0 / subset ✅ 1.0 | (自定义 scorer) | ToolCorrectness ✅ 0.8 |
| **状态一致性** | ✅ 1.0 (原生) | ✅ state_consistency 1.0 | ✅ state_consistency 1.0 | ✅ state_consistency 1.0 |
| 安全性 | 不评估 | 不评估 | ✅ yes | 不评估 |
| 响应质量 | 不评估 | LLM judge ✅ 1.0 | 不评估 | ResponseQuality ❌ 0.2 |

### 7.2 框架特性对比

| 特性 | agentevals/openevals | MLflow | DeepEval |
|------|---------------------|--------|----------|
| **安装复杂度** | 低（2 个轻量包 + langchain-aws） | 中（mlflow 较大） | 中（需要 aiobotocore） |
| **Bedrock 模型格式** | `bedrock_converse:model_id` | `bedrock:/model_id` | `AmazonBedrockModel(model=id)` |
| **轨迹匹配** | 4 种确定性匹配模式（核心优势） | 无内置 | 无内置 |
| **自定义评估** | 自定义 prompt + LLM judge | `@scorer` 装饰器 | GEval（自然语言定义） |
| **UI/可视化** | 需 LangSmith Cloud | `mlflow ui`（本地 Web） | Confident AI Cloud |
| **CI/CD 集成** | 需自行封装 | MLflow tracking | `deepeval test run`（pytest 原生） |
| **evaluate() 稳定性** | 稳定 | 内置 LLM scorer 线程卡死 | 稳定 |

### 7.3 轨迹匹配模式详解（agentevals 独有）

```
期望工具: [A, B, C, D, E]
实际工具: [B, C, D, E]

strict_match:    A,B,C,D,E == B,C,D,E ?  → ❌ (顺序+集合都要完全匹配)
unordered_match: {A,B,C,D,E} == {B,C,D,E} ?  → ❌ (集合要完全相同)
superset_match:  {B,C,D,E} ⊇ {A,B,C,D,E} ?  → ❌ (实际要包含所有期望)
subset_match:    {B,C,D,E} ⊆ {A,B,C,D,E} ?  → ✅ (实际是期望的子集)
```

适用场景：
- `strict`：流程合规性审计（如金融、医疗），必须严格按规定步骤执行
- `unordered`：确保所有关键步骤都执行了，但允许顺序灵活
- `superset`：确保 Agent 没有遗漏步骤（可以多做但不能少做）
- `subset`：确保 Agent 没有做多余的事（可以少做但不能做错）

### 7.4 评估维度选型建议

| 评估需求 | 推荐框架 | 理由 |
|----------|---------|------|
| 工具调用序列是否正确 | **agentevals** | 4 种匹配模式，无需 LLM，速度快成本零 |
| 工具调用正确性（含参数） | **DeepEval** ToolCorrectnessMetric | 内置指标，支持参数对比 |
| 回答安全性 | **MLflow** Safety | 开箱即用的安全检查 |
| 自定义业务逻辑评估 | **DeepEval** GEval 或 **MLflow** @scorer | 灵活的自然语言/代码定义 |
| CI/CD 质量门控 | **DeepEval** | pytest 原生集成，`deepeval test run` |
| 实验追踪 & 模型对比 | **MLflow** | 内置实验管理、版本追踪、Web UI |
| 轻量快速验证 | **agentevals** | 最小依赖，确定性评估无 LLM 成本 |

---

## 8. 踩坑记录与注意事项

### 8.1 模型格式差异

三个框架使用不同的模型标识格式，这是最容易出错的地方：

```python
# agentevals/openevals (基于 langchain)
model = "bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0"
# 需要 pip install langchain-aws

# MLflow (基于 litellm，但有自己的 URI 格式)
model = "bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
# 注意是 :/ 不是 /

# DeepEval (自己的模型类)
from deepeval.models import AmazonBedrockModel
model = AmazonBedrockModel(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0", region="us-east-1")
# 需要 pip install aiobotocore

# litellm (直接调用时)
model = "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0"
```

### 8.2 MLflow evaluate() 线程卡死

MLflow 3.10 的 `mlflow.genai.evaluate()` 在使用内置 LLM Scorer（Safety、Correctness）+ Bedrock 模型时会无限卡住。原因是 evaluate() 内部使用线程池并发执行 scorer，而 Bedrock 的 boto3 session 在多线程下有兼容问题。

**解决方案**：自定义 scorer 通过 `evaluate()` 执行，LLM scorer 单独直接调用：

```python
# ✅ 自定义 scorer 用 evaluate()
results = mlflow.genai.evaluate(data=data, scorers=[tool_call_accuracy])

# ✅ LLM scorer 直接调用
safety = Safety(model="bedrock:/model_id")
result = safety(outputs=text)
```

### 8.3 DeepEval GEval 的多轮对话问题

DeepEval 的 `LLMTestCase.actual_output` 只接受字符串。在多轮对话场景中，如果只传入 Agent 最后一条消息，LLM Judge 会缺少上下文，严重低估任务完成度。

**解决方案**：将完整对话历史拼接为 `actual_output`：

```python
# ❌ 只传最后一条消息
actual_output = dp.agent_output  # "Yes, that's correct! Let me confirm..."

# ✅ 拼接完整对话
full_conversation = "\n".join([
    f"{msg['role']}: {msg['content']}"
    for msg in dp.conversation
])
actual_output = full_conversation
```

### 8.4 expected_output 不能为 None

DeepEval 的 GEval 如果配置了 `LLMTestCaseParams.EXPECTED_OUTPUT`，则 `expected_output` 不能为 None，否则会抛出 `MissingTestCaseParamsError`。

```python
# ❌ 会报错
LLMTestCase(input="...", actual_output="...", expected_output=None)

# ✅ 提供默认值
expected_output = "; ".join(dp.expected_outputs) if dp.expected_outputs else "Task completed successfully"
```

### 8.5 Strands Agent state 兼容问题

新版 Strands Agents 移除了 Agent 构造函数的 `state` 参数，但工具代码中仍使用 `agent.state.get()`。需要手动挂载一个 state 对象：

```python
class _State:
    def __init__(self, data):
        self._data = data
    def get(self, key, default=None):
        return self._data.get(key, default)
    def set(self, key, value):
        self._data[key] = value

agent = Agent(model=model, system_prompt=prompt, tools=tools)
agent.state = _State({"datas": load_data()})
```

---

## 附录：项目文件结构

```
agent-evaluation/
├── main.py                         # 入口
├── run.py                          # Agent 运行逻辑
├── env_litellm.py                  # 用户模拟器
├── wiki.md                         # Agent 系统提示词
├── tools/                          # 16 个工具模块
├── data/                           # 模拟数据库
├── results/                        # 运行结果
│   ├── *_llm_0228131732.json       # 原始结果
│   ├── *_langsmith_eval.json       # agentevals 评估结果
│   ├── *_mlflow_eval.json          # MLflow 评估结果
│   └── *_deepeval_eval.json        # DeepEval 评估结果
├── eval_common/
│   └── extract_results.py          # 结果标准化提取
├── eval_langsmith/
│   └── run_eval.py                 # agentevals + openevals
├── eval_mlflow/
│   └── run_eval.py                 # MLflow GenAI
├── eval_deepeval/
│   ├── run_eval.py                 # DeepEval 脚本
│   └── test_agent.py              # pytest 风格
├── pyproject.toml                  # uv 项目配置
├── EVALUATION_GUIDE.md             # 框架对比指南
└── EVALUATION_TESTING_REPORT.md    # 本文档
```
