from typing import Any
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "get_user_details",
    "description": "Get the details of a user, including their orders.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user id, such as 'sara_doe_496'."
                }
            },
            "required": ["user_id"]
        }
    }
}

# Function name must match tool name
def get_user_details(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    user_id = tool["input"]["user_id"]
    
    datas = agent.state.get("datas") or {}
    users = datas.get("users", {})
    
    if user_id in users:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": json.dumps(users[user_id])}]
        }

    agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": "Error: user not found"}]
    }
