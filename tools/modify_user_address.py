from typing import Any
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "modify_user_address",
    "description": "Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The user id, such as 'sara_doe_496'."
                },
                "address1": {
                    "type": "string",
                    "description": "The first line of the address, such as '123 Main St'."
                },
                "address2": {
                    "type": "string",
                    "description": "The second line of the address, such as 'Apt 1' or ''."
                },
                "city": {
                    "type": "string",
                    "description": "The city, such as 'San Francisco'."
                },
                "state": {
                    "type": "string",
                    "description": "The state, such as 'CA'."
                },
                "country": {
                    "type": "string",
                    "description": "The country, such as 'USA'."
                },
                "zip": {
                    "type": "string",
                    "description": "The zip code, such as '12345'."
                }
            },
            "required": ["user_id", "address1", "address2", "city", "state", "country", "zip"]
        }
    }
}

# Function name must match tool name
def modify_user_address(tool: ToolUse, agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    user_id = tool["input"]["user_id"]
    address1 = tool["input"]["address1"]
    address2 = tool["input"]["address2"]
    city = tool["input"]["city"]
    state = tool["input"]["state"]
    country = tool["input"]["country"]
    zip_code = tool["input"]["zip"]
    
    datas = agent.state.get("datas") or {}
    users = datas.get("users", {})
    
    if user_id not in users:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: user not found"}]
        }
    
    user = users[user_id]
    user["address"] = {
        "address1": address1,
        "address2": address2,
        "city": city,
        "state": state,
        "country": country,
        "zip": zip_code,
    }
    agent.state.set("datas", datas)
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(user)}]
    }
