from typing import Any
from strands.types.tools import ToolResult, ToolUse

TOOL_SPEC = {
    "name": "calculate",
    "description": "Calculate the result of a mathematical expression.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces."
                }
            },
            "required": ["expression"]
        }
    }
}

# Function name must match tool name
def calculate(tool: ToolUse, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    expression = tool["input"]["expression"]
    
    if not all(char in "0123456789+-*/(). " for char in expression):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: invalid characters in expression"}]
        }
    
    try:
        # Evaluate the mathematical expression safely
        result = str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}]
        }
    except Exception as e:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": f"Error: {e}"}]
        }
