from typing import Any
from strands.types.tools import ToolResult, ToolUse
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "find_user_id_by_name_zip",
    "description": "Find user id by first name, last name, and zip code. If the user is not found, the function will return an error message. By default, find user id by email, and only call this function if the user is not found by email or cannot remember email.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "first_name": {
                    "type": "string",
                    "description": "The first name of the customer, such as 'John'."
                },
                "last_name": {
                    "type": "string",
                    "description": "The last name of the customer, such as 'Doe'."
                },
                "zip": {
                    "type": "string",
                    "description": "The zip code of the customer, such as '12345'."
                }
            },
            "required": ["first_name", "last_name", "zip"]
        }
    }
}

# Function name must match tool name
def find_user_id_by_name_zip(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    first_name = tool["input"]["first_name"]
    last_name = tool["input"]["last_name"]
    zip_code = tool["input"]["zip"]
    
    datas = agent.state.get("datas") or {}
    users = datas.get("users", {})
    
    for user_id, profile in users.items():
        if (profile["name"]["first_name"].lower() == first_name.lower() and
            profile["name"]["last_name"].lower() == last_name.lower() and
            profile["address"]["zip"] == zip_code):
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": user_id}]
            }
    
    agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": "Error: user not found"}]
    }
