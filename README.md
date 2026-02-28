# Agent Evaluation - 零售客服智能体评估系统

基于 tau-bench 框架的智能体评估系统，专门用于评估零售客服智能体在处理客户服务任务时的性能表现。同时集成了 MLflow、LangSmith/agentevals、DeepEval 三个评估框架进行对比评估。

## 项目概述

本项目实现了一个完整的智能体评估环境，模拟真实的零售客服场景，包括订单管理、用户认证、产品查询、退换货处理等功能。系统使用 AWS Bedrock 上的 Claude 模型作为智能体和用户模拟器。

## 主要功能

### 智能体能力
- **用户身份认证**: 通过邮箱或姓名+邮编验证用户身份
- **订单管理**: 查询、取消、修改待处理订单
- **退换货处理**: 处理已送达订单的退货和换货请求
- **用户信息管理**: 查询和修改用户地址信息
- **产品信息查询**: 提供产品详情和类型列表
- **人工转接**: 在无法处理的情况下转接人工客服

### 多框架评估
- **tau-bench**: 数据库状态 hash 对比，判断任务是否完成
- **MLflow**: 内置 Safety/Correctness LLM Judge + 自定义 @scorer
- **agentevals/openevals**: 4种轨迹匹配模式 + LLM-as-Judge
- **DeepEval**: ToolCorrectnessMetric + GEval 自定义指标 + pytest 集成

### 两种评估范式
- **轨迹评估**: 检查工具调用序列是否匹配、LLM 判断输出质量（三个框架均支持）
- **状态一致性评估**: 不关心路径，只看最终数据库状态是否与 golden state 一致（`state_consistency` 指标，确定性、无需 LLM）

## 技术架构

### 核心组件
- **Agent**: 基于 Strands 框架的智能体实现
- **Environment**: 任务执行环境和奖励计算（env_litellm.py）
- **Tools**: 16个专用工具函数，涵盖所有客服场景
- **Data**: JSON 格式的模拟数据库
- **Evaluation**: 三个评估框架 + 公共数据提取层（含状态一致性评估）

### 技术栈
- **运行时**: Python 3.13+
- **包管理**: uv
- **Agent 框架**: Strands Agents
- **模型**: AWS Bedrock (Claude)
- **监控**: Langfuse 遥测和追踪

## 快速开始

### 环境要求

```bash
# 克隆项目
git clone https://github.com/xqun3/agent-evaluation.git && cd agent-evaluation

# 克隆 tau-bench
git clone https://github.com/sierra-research/tau-bench

# 安装所有依赖（推荐 uv）
uv sync

# 或使用 pip
pip install -r requirements.txt
pip install -e ./tau-bench
```

### AWS 配置

在 `.env` 文件中配置 AWS 凭证：

```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION_NAME=us-east-1
```

### 运行 Agent

```bash
# 单条任务测试
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

# 全量测试
uv run python main.py \
  --agent-strategy tool-calling \
  --env retail \
  --model us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --model-provider bedrock \
  --user-model us.anthropic.claude-3-5-haiku-20241022-v1:0 \
  --user-model-provider bedrock \
  --user-strategy llm \
  --max-concurrency 1
```

### 运行评估

```bash
# LangSmith/agentevals（含轨迹匹配 + LLM Judge）
uv run python eval_langsmith/run_eval.py results/<your_results>.json

# MLflow（含 Safety/Correctness + 自定义 scorer）
uv run python eval_mlflow/run_eval.py results/<your_results>.json

# DeepEval（含 ToolCorrectness + GEval）
uv run python eval_deepeval/run_eval.py results/<your_results>.json

# DeepEval pytest 风格
RESULTS_PATH=results/<your_results>.json uv run deepeval test run eval_deepeval/test_agent.py
```

### LLM as judge 进行归因分析

```bash
uv run python auto_error_identification.py \
  --env retail \
  --platform bedrock \
  --results-path <your_results>.json \
  --output-path test-auto-error-identification
```

## 文档

- [EVALUATION_GUIDE.md](./EVALUATION_GUIDE.md) — 三个评估框架的对比指南、选型建议
- [EVALUATION_TESTING_REPORT.md](./EVALUATION_TESTING_REPORT.md) — 完整测试过程记录、输入输出示例、踩坑记录

## 致谢

本项目基于 [tau-bench](https://github.com/sierra-research/tau-bench) 开源项目构建。感谢 Sierra Research 团队提供的优秀智能体评估框架，tau-bench 是一个 Agent 评估基准测试框架，提供了标准化的评估环境和指标体系，极大地简化了智能体性能评估的复杂性。
