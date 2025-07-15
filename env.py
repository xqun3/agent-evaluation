# Copyright Sierra

import random
import copy
import json
import uuid
from typing import List, Optional

import boto3
from tau_bench.types import SolveResult
from tau_bench.types import (
    Action,
    Task,
    EnvInfo,
    EnvResetResponse,
    RewardResult,
    RewardOutputInfo,
    RewardActionInfo,
)

from tools import get_tool_by_name
from data import load_data
from utils import generate_conversation, get_data_hash
# from strands import Agent

# ToHashable = Union[
#     str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
# ]
# Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]


# def to_hashable(item: ToHashable) -> Hashable:
#     if isinstance(item, dict):
#         return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
#     elif isinstance(item, list):
#         return tuple(to_hashable(element) for element in item)
#     elif isinstance(item, set):
#         return tuple(sorted(to_hashable(element) for element in item))
#     else:
#         return item


# def consistent_hash(
#     value: Hashable,
# ) -> str:
#     return sha256(str(value).encode("utf-8")).hexdigest()


class Env(object):
    def __init__(
        self,
        tasks: List[Task],
        agent,
        terminate_tools: List[str] = [],
        task_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.terminate_tools = terminate_tools
        self.tasks = tasks
        if task_index is not None:
            self.task_index = task_index
        else:
            self.task_index = random.randint(0, len(tasks))
        self.task = tasks[self.task_index]
        random_uuid = uuid.uuid4()
        print(str(random_uuid))  

        # boto_config = BotocoreConfig(
        #     retries={"max_attempts": 3, "mode": "standard"},
        #     connect_timeout=5,
        #     read_timeout=60
        # )

        # Create a configured Bedrock model
        # model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        # bedrock_model = BedrockModel(
        #     model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0",
        #     region_name="us-east-1",  # Specify a different region than the default
        #     temperature=0.0,
        #     boto_client_config=boto_config,
        # )
        # self.user_agent = Agent(
        #             model = bedrock_model,
        #             system_prompt=self.build_user_system_prompt(self.task.instruction),
        #             trace_attributes={
        #                 "session.id": f"tao-bench-user-{str(random_uuid)}",
        #                 "user.id": "xq",
        #                 "langfuse.tags": [
        #                     "evaluation",
        #                     "retail-user",
        #                 ],
        #                 "encoding": "utf-8"  # 明确指定编码
        #             }) 
        self.user_system_prompt = self.build_user_system_prompt(self.task.instruction)
        self.client = boto3.client(service_name='bedrock-runtime') 
        self.user_messages = []
        self.actions: List[Action] = []
        self.output_list: List[str] = []
        self.split_message_ids = 0
        # self.model_id = "us.anthropic.claude-3-5-haiku-20241022-v1:0" 
        self.model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is None:
            task_index = random.randint(0, len(self.tasks))
        self.task_index = task_index
        # self.data = self.data_load_func()
        self.task = self.tasks[task_index]
        self.actions = []
        
    def build_user_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent.{instruction_display}
Rules:
- Just generate one line at a time to simulate the user's message.
- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

    def loop(self, max_num_steps=30):
        # total_cost = 0.0
        accumulated_usage = {
            "inputTokens":0,
            "outputTokens": 0,
            "totalTokens": 0}
        # env_reset_res = env.reset(task_index=task_index)
        # user_message = self.user_agent("Hi! How can I help you today?")
        self.user_messages = [{"role": "user","content": [{"text": "Hi! How can I help you today?"}]}]
        # user_message = None
        user_message = generate_conversation(self.client, self.model_id, self.user_messages, system_prompt=self.user_system_prompt,max_token=8192) 
        
        for _ in range(max_num_steps):
            reward = 0
            done = False
            done = "###STOP###" in f"{user_message}"
            self.user_messages.append({"role": "assistant","content": [{"text": user_message}]})
            if done:
                self.split_message_ids = len(self.agent.messages)
                if hasattr(res, 'metrics') and hasattr(res.metrics, 'tool_metrics'):
                    for tool_name, tool_metric in res.metrics.tool_metrics.items():
                        if tool_metric.call_count > 0:
                            self.actions.append(Action(
                                name=tool_metric.tool["name"],
                                kwargs=tool_metric.tool["input"],
                            ))
                reward_res = self.calculate_reward()
                reward = reward_res.reward
                info.reward_info = reward_res
                break 
            user_input = f"{user_message}"
            res = self.agent(user_input)
            accumulated_usage["inputTokens"] += res.metrics.accumulated_usage["inputTokens"]
            accumulated_usage["outputTokens"] += res.metrics.accumulated_usage["outputTokens"]
            accumulated_usage["totalTokens"] += res.metrics.accumulated_usage['totalTokens']

            agent_output = f"{res}"
            self.user_messages.append({"role": "user","content": [{"text": agent_output}]})
            user_message = generate_conversation(self.client, self.model_id, self.user_messages, system_prompt=self.user_system_prompt,max_token=8192) 
            self.output_list.append(agent_output)

            info = EnvInfo(task=self.task)
            if len(self.agent.messages) > 3: 
                for dic in self.agent.messages[-3]["content"]:
                    if "toolUse" in dic and dic["toolUse"]["name"] in self.terminate_tools:
                        done = True


            if done:
                self.split_message_ids = len(self.agent.messages)
                if hasattr(res, 'metrics') and hasattr(res.metrics, 'tool_metrics'):
                    for tool_name, tool_metric in res.metrics.tool_metrics.items():
                        if tool_metric.call_count > 0:
                            self.actions.append(tool_name)
                            self.actions.append(Action(
                                name=tool_metric.tool["name"],
                                kwargs=tool_metric.tool["input"],
                            ))
                reward_res = self.calculate_reward()
                reward = reward_res.reward
                info.reward_info = reward_res
                break
        
        total_cost = accumulated_usage["inputTokens"] /1000 * 0.003 + accumulated_usage["outputTokens"]/1000 * 0.015
        print("total_cost: ", total_cost)
        # info.total
        final_messages = self.agent.messages[:self.split_message_ids]
        final_messages.append({"role": "user","content": [{"text": user_message}]})
        return SolveResult(
            reward=reward,
            info=info.model_dump(),
            messages=final_messages,
            total_cost=total_cost,
        )

    def calculate_reward(self) -> RewardResult:
        reward = 1.0

        # Check if the database changes are correct. If they are not correct, then we set the reward to 0.
        # TODO: cache gt_data_hash in tasks.py (low priority)

        data_hash = get_data_hash(self.agent.state.get("datas"))
        # with open("run_result_data","w") as wf:
        #     json.dump(self.agent.state.get("datas"), wf)
        golden_data = load_data()
        actions = []

        for action in self.task.actions:
            if action.name not in self.terminate_tools:
                actions.append(action)
                parameters = copy.deepcopy(action.kwargs)
                tool_use = {
                    "toolUseId": "123",
                    "input": parameters
                }
                print(parameters.keys())
                tool_func = get_tool_by_name(action.name)
                _ = tool_func(tool_use, agent=None, datas=golden_data)
                print(_)
        # with open("golden_data","w") as wf:
        #     json.dump(golden_data, wf)
                 
        gt_data_hash = get_data_hash(golden_data)
        info = RewardActionInfo(
            r_actions=data_hash == gt_data_hash, gt_data_hash=gt_data_hash
        )
        if not info.r_actions:
            reward = 0.0

        if len(self.task.outputs) > 0:
            r_outputs = 1.0
            outputs = {}
            for output in self.task.outputs:
                found = False
                for res in self.output_list:
                    if (
                        output.lower()
                        in res.lower().replace(",", "")
                    ):
                        found = True
                        break
                outputs[output] = found
                if not found:
                    r_outputs = 0.0
                    reward = 0.0
            info = RewardOutputInfo(r_outputs=r_outputs, outputs=outputs)
            
        return RewardResult(reward=reward, info=info, actions=self.actions)
# 