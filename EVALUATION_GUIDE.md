# Agent 评估框架对比指南

本文档基于对 tau-bench 零售客服 Agent 的实际评估体验，对比了 MLflow、LangSmith/agentevals 和 DeepEval 三个开源评估框架。

> 完整的测试过程记录、输入输出示例和踩坑记录请参阅 [EVALUATION_TESTING_REPORT.md](./EVALUATION_TESTING_REPORT.md)。

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv（推荐）
git clone https://github.com/sierra-research/tau-bench
uv sync

# 或使用 pip
pip install -r requirements.txt
pip install -e ./tau-bench
```

### 2. 运行 Agent 获取结果

```bash
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
```

### 3. 运行三个评估框架

```bash
# LangSmith/agentevals 评估
uv run python eval_langsmith/run_eval.py results/<your_results>.json

# MLflow 评估
uv run python eval_mlflow/run_eval.py results/<your_results>.json

# DeepEval 评估
uv run python eval_deepeval/run_eval.py results/<your_results>.json

# DeepEval pytest 风格
RESULTS_PATH=results/<your_results>.json uv run deepeval test run eval_deepeval/test_agent.py
```

---

## 框架对比表

| 特性 | MLflow | LangSmith/agentevals | DeepEval |
|------|--------|---------------------|----------|
| **安装** | `uv add 'mlflow[genai]>=3.3'` | `uv add openevals agentevals langchain-aws` | `uv add deepeval aiobotocore` |
| **工具调用评估** | 自定义 @scorer | trajectory_match (4种模式) | ToolCorrectnessMetric |
| **LLM-as-Judge** | 内置 Correctness/Safety | create_llm_as_judge | GEval (自定义) |
| **轨迹评估** | 无内置 | trajectory_strict/unordered/subset/superset_match | 无内置 |
| **自定义指标** | @scorer 装饰器 | create_llm_as_judge + 自定义 prompt | GEval (自然语言定义) |
| **UI/可视化** | MLflow UI (本地) | LangSmith Cloud | Confident AI Cloud |
| **pytest 集成** | 无 | 无 | 原生支持 (deepeval test run) |
| **数据格式** | Dict/DataFrame | OpenAI message format | LLMTestCase/ToolCall |
| **Bedrock 模型格式** | `bedrock:/model_id` | `bedrock_converse:model_id` | `AmazonBedrockModel(model=id)` |
| **CI/CD 集成** | MLflow tracking | 需自行封装 | pytest + deepeval |
| **状态一致性评估** | @scorer state_consistency | run_state_consistency_eval() | 独立计算 + summary 输出 |

---

## 各框架详细说明

### MLflow (`eval_mlflow/`)

#### 优势
- **一站式 ML 平台**：评估结果与模型版本、实验追踪紧密集成
- **丰富的内置 Scorer**：20+ 内置评估指标，覆盖正确性、安全性等
- **MLflow UI**：本地 Web UI 查看评估结果，便于对比不同实验
- **生产就绪**：适合从开发到生产的完整 ML 生命周期

#### 劣势
- **依赖较重**：安装包较大，引入大量依赖
- **evaluate() 线程问题**：内置 LLM Scorer 在 `evaluate()` 中使用 Bedrock 时会卡死，需单独直接调用
- **学习曲线**：概念较多（Experiment, Run, Trace, Scorer）

#### 核心代码示例

```python
import mlflow
from mlflow.genai.scorers import Correctness, Safety
from mlflow.genai import scorer
from mlflow.genai.scorers.base import Feedback  # 注意：Feedback 在 base 子模块中

# 自定义 scorer — 通过 evaluate() 执行
@scorer(name="tool_call_accuracy")
def tool_call_accuracy(inputs, outputs, expectations):
    expected = set(expectations.get("expected_tool_names", []))
    actual = set(outputs.get("tool_names", []))
    score = len(expected & actual) / len(expected) if expected else 1.0
    return Feedback(score=score, rationale=f"Matched {expected & actual}")

# 自定义 scorer 通过 evaluate() 执行
results = mlflow.genai.evaluate(
    data=eval_data,
    scorers=[tool_call_accuracy],  # 仅自定义 scorer
)

# 内置 LLM Scorer 需单独直接调用（避免 evaluate 线程卡死）
# 注意模型格式为 provider:/model_name
safety = Safety(model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0")
result = safety(outputs="Agent response text")

correctness = Correctness(model="bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0")
result = correctness(
    inputs={"question": "user query"},
    outputs="Agent response",
    expectations={"expected_response": "expected answer"},
)
```

---

### LangSmith/agentevals (`eval_langsmith/`)

#### 优势
- **轨迹匹配灵活**：4种匹配模式（strict/unordered/superset/subset）精确评估工具调用序列
- **轻量级**：openevals 和 agentevals 包小巧，安装快速
- **标准格式**：使用 OpenAI message format，兼容性强
- **确定性评估**：轨迹匹配无需 LLM，速度快、成本零

#### 劣势
- **无本地 UI**：结果展示需要 LangSmith Cloud 或自行处理
- **LLM Judge 依赖**：轨迹 LLM-as-Judge 和 correctness 评估需要 LLM API
- **额外依赖**：使用 Bedrock 需要安装 `langchain-aws`

#### 核心代码示例

```python
from agentevals.trajectory.match import create_trajectory_match_evaluator
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

# 轨迹匹配 - 不需要 LLM，速度快
evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered",   # strict / unordered / superset / subset
    tool_args_match_mode="ignore",       # 只匹配工具名
)
result = evaluator(outputs=actual_traj, reference_outputs=expected_traj)

# LLM-as-Judge 正确性
# 注意模型格式为 provider:model_name（基于 langchain，需 langchain-aws）
correctness = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0",
)
result = correctness(inputs={"question": q}, outputs={"answer": a})
```

---

### DeepEval (`eval_deepeval/`)

#### 优势
- **pytest 原生集成**：`deepeval test run` 直接用 pytest 风格写评估，适合 CI/CD
- **GEval 灵活性高**：用自然语言定义评估标准，无需写复杂代码
- **ToolCall 数据结构**：原生支持工具调用评估，数据模型清晰
- **Bedrock 原生支持**：`AmazonBedrockModel` 开箱即用

#### 劣势
- **LLM 依赖较强**：大部分指标需要 LLM 调用，成本较高
- **多轮对话评估有局限**：`actual_output` 只接受字符串，需要自行拼接完整对话历史
- **Cloud 依赖**：部分高级功能需要 Confident AI 账号

#### 核心代码示例

```python
from deepeval import assert_test, evaluate
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ToolCorrectnessMetric, GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AmazonBedrockModel

# 使用 Bedrock 模型（需 aiobotocore）
bedrock_model = AmazonBedrockModel(
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    region="us-east-1",
)

test_case = LLMTestCase(
    input="帮我取消订单 #W001",
    actual_output="已为您取消订单 #W001，退款将在3-5个工作日到账。",
    expected_output="Task completed successfully",  # 不能为 None
    tools_called=[ToolCall(name="cancel_pending_order", input_parameters={"order_id": "#W001"})],
    expected_tools=[ToolCall(name="cancel_pending_order")],
)

# pytest 风格
assert_test(test_case, [
    ToolCorrectnessMetric(threshold=0.5, model=bedrock_model),
    GEval(name="TaskCompletion", evaluation_steps=[...], threshold=0.5, model=bedrock_model),
])

# 或批量评估
results = evaluate(test_cases=[test_case], metrics=[...])
```

---

## 适用场景推荐

### 场景 1：快速验证 Agent 工具调用正确性
**推荐：agentevals**
- 轨迹匹配评估无需 LLM，速度快、成本零
- 4种匹配模式覆盖不同严格程度的需求

### 场景 2：CI/CD 集成的 Agent 质量门控
**推荐：DeepEval**
- pytest 原生集成，`deepeval test run` 可直接嵌入 CI pipeline
- 支持阈值设定，自动 pass/fail 判定

### 场景 3：全面的实验追踪与模型对比
**推荐：MLflow**
- 实验追踪、版本管理、结果可视化一站式解决
- 适合在多模型、多策略间做系统性对比

### 场景 4：自定义业务评估标准
**推荐：DeepEval GEval 或 MLflow @scorer**
- GEval 允许用自然语言定义评估标准
- MLflow @scorer 提供更灵活的编程式自定义

### 场景 5：验证 Agent 最终效果而非路径（状态一致性评估）
**推荐：三个框架均支持 state_consistency 指标**
- 不关心 agent 走了什么路径，只看最终数据库状态是否正确
- 确定性评估，无需 LLM，速度快、成本零
- 与 tau-bench 原生评估思路一致：通过数据库状态 hash 对比判断任务完成度
- 适用于允许多条正确路径的场景（如 agent 跳过了某步骤但最终结果仍正确）

---

## 重要注意事项

### Bedrock 模型格式差异

三个框架使用不同的模型标识格式：

```python
# agentevals/openevals (基于 langchain)
"bedrock_converse:global.anthropic.claude-sonnet-4-5-20250929-v1:0"

# MLflow (自有 URI 格式，注意是 :/ 不是 /)
"bedrock:/global.anthropic.claude-sonnet-4-5-20250929-v1:0"

# DeepEval (自有模型类)
AmazonBedrockModel(model="global.anthropic.claude-sonnet-4-5-20250929-v1:0", region="us-east-1")
```

### MLflow evaluate() 线程卡死

MLflow 3.10 的 `mlflow.genai.evaluate()` 在使用内置 LLM Scorer + Bedrock 时会卡住。解决方案：自定义 scorer 通过 `evaluate()` 执行，LLM scorer 单独直接调用。

### DeepEval 多轮对话上下文

DeepEval 的 `actual_output` 只包含 Agent 最后一条消息。在多轮对话场景中，需要将完整对话历史拼接后传入，否则 GEval 会低估任务完成度。

> 更多踩坑详情请参阅 [EVALUATION_TESTING_REPORT.md](./EVALUATION_TESTING_REPORT.md#8-踩坑记录与注意事项)。

---

## 评估数据流

```
Agent 运行 (run.py)
    ↓
results/*.json (原始对话轨迹 + 工具调用 + 奖励)
    ↓
eval_common/extract_results.py (标准化提取)
    ↓                  ↓
    ↓            eval_common/state_eval.py (重放工具调用 → hash 对比)
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│  eval_mlflow/   │  eval_langsmith/ │  eval_deepeval/  │
│  run_eval.py    │  run_eval.py     │  run_eval.py     │
│                 │                  │  test_agent.py   │
└─────────────────┴──────────────────┴─────────────────┘
    ↓                   ↓                   ↓
 MLflow UI         JSON report      pytest report /
                                    Confident AI
```

> 详细的 eval_common 使用说明见 [eval_common/README.md](./eval_common/README.md)。

---

## 项目结构

```
agent-evaluation/
├── main.py                         # 入口：CLI 参数解析
├── run.py                          # 核心：Agent 运行逻辑
├── env_litellm.py                  # 环境：用户模拟器 + 评估循环
├── wiki.md                         # Agent 系统提示词
├── tools/                          # 16 个工具模块
├── data/                           # 模拟数据库
├── results/                        # 运行结果输出
├── eval_common/                    # 公共评估模块
│   ├── state_eval.py               # 通用状态一致性评估（MockAgent + replay）
│   ├── extract_results.py          # 结果提取与标准化
│   └── README.md                   # eval_common 使用文档
├── eval_mlflow/                    # MLflow 评估
│   └── run_eval.py
├── eval_langsmith/                 # LangSmith/agentevals 评估
│   └── run_eval.py
├── eval_deepeval/                  # DeepEval 评估
│   ├── run_eval.py
│   └── test_agent.py              # pytest 风格测试
├── pyproject.toml                  # uv 项目配置 + 依赖
├── requirements.txt                # pip 依赖清单
├── EVALUATION_GUIDE.md             # 本文档（框架对比指南）
└── EVALUATION_TESTING_REPORT.md    # 测试报告（完整输入输出示例）
```

---

## 总结

| 维度 | MLflow | agentevals/openevals | DeepEval |
|------|--------|---------------------|----------|
| **上手难度** | 中等 | 低 | 低 |
| **评估深度** | 高 | 中-高 | 高 |
| **工具调用** | 需自定义 | 最佳（4种轨迹匹配） | 好（内置 ToolCorrectness） |
| **CI/CD** | 一般 | 需自建 | 最佳（pytest 原生） |
| **可视化** | 最佳 (本地UI) | 需 Cloud | 需 Cloud |
| **成本** | 中 (LLM judge) | 低-中 | 中 (LLM judge) |
| **Bedrock 兼容性** | 有线程问题 | 稳定 | 稳定 |
| **生态集成** | ML 生态 | LangChain 生态 | 独立 |
| **状态评估** | @scorer (确定性) | 确定性函数 | 独立计算 |

**建议**：根据团队技术栈和需求选择。如果已使用 MLflow 管理模型，选 MLflow；如果需要精确的轨迹匹配，选 agentevals；如果需要 CI/CD 集成，选 DeepEval。三者可以互补使用。
