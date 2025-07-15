from typing import Any
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "think",
    "description": "Use the tool to think about something. It will not obtain new information or change the database, but just append the thought to the log. Use it when complex reasoning or some cache memory is needed.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": "A thought to think about."
                }
            },
            "required": ["thought"]
        }
    }
}

# Function name must match tool name
def think(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    
    # This method does not change the state of the data; it simply returns an empty string.
    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": ""}]
    }
