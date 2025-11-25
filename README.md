# Agent Evaluation - 零售客服智能体评估系统

基于 tau-bench 框架的智能体评估系统，专门用于评估零售客服智能体在处理客户服务任务时的性能表现。

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

### 评估环境
- **任务驱动**: 基于预定义任务集进行评估
- **用户模拟**: 自动生成用户对话模拟真实交互
- **性能指标**: 计算成功率和 Pass@k 指标
- **并发支持**: 支持多任务并行执行
- **结果追踪**: 详细记录执行轨迹和成本

## 技术架构

### 核心组件
- **Agent**: 基于 Strands 框架的智能体实现
- **Environment**: 任务执行环境和奖励计算
- **Tools**: 17个专用工具函数，涵盖所有客服场景
- **Data**: 模拟的用户、产品、订单数据库
- **Evaluation**: 自动化评估和指标计算

### 技术栈
- **框架**: Strands Agent Framework
- **监控**: Langfuse 遥测和追踪
- **数据**: JSON 格式的模拟数据库
- **语言**: Python 3.8+

## 快速开始

### 环境要求
```bash
git clone https://github.com/sierra-research/tau-bench && cd ./tau-bench
pip install -e .
cd ../
git clone https://github.com/xqun3/agent-evaluation.git && cd agent-evaluation
pip install -r requirements.txt
```

### AWS 配置
确保已配置 AWS 凭证和 Bedrock 访问权限：
```bash
aws configure
```

### 运行测试任务
```
 python main.py --agent-strategy tool-calling --env retail --model bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0 --model-provider bedrock --user-model bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0 --user-model-provider bedrock --user-strategy llm --max-concurrency 1
```

### LLM as judge 进行归因分析
```
python auto_error_identification.py --env retail --platform bedrock --results-path <Your-output-result.json>  --output-path test-auto-error-identification
```

## 致谢

本项目基于 [tau-bench](https://github.com/sierra-research/tau-bench) 开源项目构建。感谢 Sierra Research 团队提供的优秀智能体评估框架，tau-bench 是一个 Agent 评估基准测试框架，提供了标准化的评估环境和指标体系，极大地简化了智能体性能评估的复杂性。

