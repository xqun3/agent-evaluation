from typing import Any
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "transfer_to_human_agents",
    "description": "Transfer the user to a human agent, with a summary of the user's issue. Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A summary of the user's issue."
                }
            },
            "required": ["summary"]
        }
    }
}

# Function name must match tool name
def transfer_to_human_agents(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    
    # This method simulates the transfer to a human agent.
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": "Transfer successful"}]
    }
