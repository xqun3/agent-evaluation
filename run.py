
import base64
import json
import os
import sys
from typing import List 
import time
import random
import traceback
from typing import List
from datetime import datetime
from math import comb
import multiprocessing
from dotenv import load_dotenv

import boto3
from strands.telemetry import StrandsTelemetry
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands import Agent
from tau_bench.types import EnvRunResult, RunConfig
from botocore.config import Config as BotocoreConfig

from tools import ALL_TOOLS
# Initialize global variable
tool_modules = []
for tool in ALL_TOOLS:
    # 获取函数所在的模块
    module_name = tool.__module__
    module = sys.modules[module_name]
    tool_modules.append(module)

# from env import Env
from env_litellm import Env
from data import load_data


# 修改 JSON 序列化方式，确保中文正确显示
# 通过猴子补丁替换 json.dumps 方法，确保所有 JSON 序列化都使用 ensure_ascii=False
original_dumps = json.dumps
def custom_dumps(*args, **kwargs):
    kwargs['ensure_ascii'] = False
    return original_dumps(*args, **kwargs)
json.dumps = custom_dumps


load_dotenv(".env", override=True)
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# AWS_REGION_NAME = os.getenv("AWS_REGION_NAME")
# print(f"AKSK: {AWS_ACCESS_KEY_ID} {AWS_SECRET_ACCESS_KEY}")

# os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
# os.environ["LANGFUSE_SECRET_KEY"] = secret_key
# # os.environ["LANGFUSE_HOST"] = "http://localhost:3000" # 🇪🇺 EU region (default)
# os.environ["LANGFUSE_HOST"] = langfuse_endpoint
 
public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
langfuse_endpoint =  os.environ.get("LANGFUSE_HOST")
# Set up endpoint
if public_key and secret_key and langfuse_endpoint:
    otel_endpoint = langfuse_endpoint + "/api/public/otel"
    auth_token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = otel_endpoint
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {auth_token}"
    strands_telemetry = StrandsTelemetry()
    strands_telemetry.setup_otlp_exporter()


def run(config: RunConfig) -> List[EnvRunResult]:
    assert config.env in ["retail", "airline"], "Only retail and airline envs are supported"
    assert config.task_split in ["train", "test", "dev"], "Invalid task split"

    random.seed(config.seed)
    time_str = datetime.now().strftime("%m%d%H%M%S")
    ckpt_path = f"{config.log_dir}/{config.agent_strategy}-{config.model.split('/')[-1]}-{config.temperature}_range_{config.start_index}-{config.end_index}_user-{config.user_model.split('/')[-1]}-{config.user_strategy}_{time_str}.json"
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)

    with open(os.path.join("./", "wiki.md"), "r") as f:
        system_prompt = f.read()

    match config.task_split:
        case "test":
            from tau_bench.envs.retail.tasks_test import TASKS_TEST as tasks
        case "train":
            from tau_bench.envs.retail.tasks_train import TASKS_TRAIN as tasks
        case "dev":
            from tau_bench.envs.retail.tasks_dev import TASKS_DEV as tasks
        case _:
            raise ValueError(f"Unknown task split: {config.task_split}")

    end_index = (
        len(tasks) if config.end_index == -1 else min(config.end_index, len(tasks))
    )
    results: List[EnvRunResult] = []
    lock = multiprocessing.Lock()
    if config.task_ids and len(config.task_ids) > 0:
        print(f"Running tasks {config.task_ids} (checkpoint path: {ckpt_path})")
    else:
        print(
            f"Running tasks {config.start_index} to {end_index} (checkpoint path: {ckpt_path})"
    )

    boto_config = BotocoreConfig(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60
    )

    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME
    )

    # model = BedrockModel(
    #     model_id=config.model,
    #     boto_session=session,
    #     # region_name="us-west-2",  # Specify a different region than the default
    #     temperature=config.temperature,
    #     boto_client_config=boto_config,
    # )
    model = OpenAIModel(
        client_args={
            "base_url": f"http://test-model-e051c6f0f76ab9cf.elb.us-east-2.amazonaws.com:80/v1", 
            "api_key": "None"
            # "api_key": "<KEY>",
        },
        # **model_config
        model_id=config.model,
        params={
            "temperature": config.temperature,
        }
    )
    total_cost = 0
    for i in range(config.num_trials):
        if config.task_ids and len(config.task_ids) > 0:
            idxs = config.task_ids
        else:
            idxs = list(range(config.start_index, end_index))
        if config.shuffle:
            random.shuffle(idxs)

        def _run(idx: int, total_cost: int) -> EnvRunResult:
            print(f"Running task {idx}")
            try:
                agent = Agent(model=model,
                    system_prompt=system_prompt,
                    tools= tool_modules,
                    state={"datas": load_data()},
                    trace_attributes={
                        "session.id": f"test-retail-{idx}-{config.model.split('/')[-1]}",
                        "user.id": f"agent-{idx}-{config.model.split('/')[-1]}",
                        "langfuse.tags": [
                            f"retail-agent-{idx}-{config.model.split('/')[-1]}",
                        ],
                        "encoding": "utf-8"  # 明确指定编码
                    })
                env = Env(tasks, agent, ["transfer_to_human_agents"], idx, config)
                env.reset(idx)
                res = env.loop()
                total_cost += res.total_cost 
                result = EnvRunResult(
                    task_id=idx,
                    reward=res.reward,
                    info=res.info,
                    traj=res.messages,
                    trial=i,
                )
            except Exception as e:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(e), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=i,
                )
                        

            print(
                "✅" if result.reward == 1 else "❌",
                f"task_id={idx}",
                result.info,
            )
            print("-----")
            with lock:
                data = []
                if os.path.exists(ckpt_path):
                    with open(ckpt_path, "r") as f:
                        data = json.load(f)
                with open(ckpt_path, "w") as f:
                    json.dump(data + [result.model_dump()], f, indent=2)
            return result, total_cost
        # _run(0)
        for idx in idxs:
            result, total_cost = _run(idx, total_cost) 
            results.append(result)
            time.sleep(60)

        print("total_cost: ", total_cost)

    display_metrics(results, config.num_trials)

    with open(ckpt_path, "w") as f:
        json.dump([result.model_dump() for result in results], f, indent=2)
        print(f"\n📄 Results saved to {ckpt_path}\n")
    return results


def display_metrics(results: List[EnvRunResult], num_trials) -> None:
    def is_successful(reward: float) -> bool:
        return (1 - 1e-6) <= reward <= (1 + 1e-6)
    # print(results)

    # num_trials = len(set([r.trial for r in results]))
    rewards = [r.reward for r in results]
    avg_reward = sum(rewards) / len(rewards)
    # c from https://arxiv.org/pdf/2406.12045
    c_per_task_id: dict[int, int] = {}
    for result in results:
        # print(result.trial)
        if result.task_id not in c_per_task_id:
            c_per_task_id[result.task_id] = 1 if is_successful(result.reward) else 0
        else:
            c_per_task_id[result.task_id] += 1 if is_successful(result.reward) else 0
    pass_hat_ks: dict[int, float] = {}
    for k in range(1, num_trials + 1):
        sum_task_pass_hat_k = 0
        for c in c_per_task_id.values():
            sum_task_pass_hat_k += comb(c, k) / comb(num_trials, k)
        pass_hat_ks[k] = sum_task_pass_hat_k / len(c_per_task_id)
    print(f"🏆 Average reward: {avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")


