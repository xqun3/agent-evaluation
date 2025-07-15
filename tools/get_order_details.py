from typing import Any
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "get_order_details",
    "description": "Get the status and details of an order.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
                }
            },
            "required": ["order_id"]
        }
    }
}

# Function name must match tool name
def get_order_details(tool: ToolUse, agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    order_id = tool["input"]["order_id"]
    
    print(tool_use_id, order_id)
    if "datas" in kwargs:
        datas = kwargs["datas"]
    else:   
        datas = agent.state.get("datas") or {} 
    orders = datas.get("orders", {})
    
    if order_id in orders:
        result = json.dumps(orders[order_id])
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": result}]
        }
    
    if agent is not None:
        agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": "Error: order not found"}]
    }
