from typing import Any
from strands.types.tools import ToolResult, ToolUse
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "find_user_id_by_email",
    "description": "Find user id by email. If the user is not found, the function will return an error message.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The email of the user, such as 'something@example.com'."
                }
            },
            "required": ["email"]
        }
    }
}

# Function name must match tool name
def find_user_id_by_email(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    email = tool["input"]["email"]
    
    if "datas" in kwargs:
        datas = kwargs["datas"]
    else:   
        datas = agent.state.get("datas") or {} 
    users = datas.get("users", {})
    
    for user_id, profile in users.items():
        if profile["email"].lower() == email.lower():
            return {
                "toolUseId": tool_use_id,
                "status": "success",
                "content": [{"text": user_id}]
            }
    

    if agent is not None:
        agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": "Error: user not found"}]
    }
