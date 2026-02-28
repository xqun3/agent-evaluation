from typing import Any
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "get_product_details",
    "description": "Get the inventory details of a product.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The product id, such as '6086499569'. Be careful the product id is different from the item id."
                }
            },
            "required": ["product_id"]
        }
    }
}

# Function name must match tool name
def get_product_details(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    product_id = tool["input"]["product_id"]
    
    datas = agent.state.get("datas") or {}
    products = datas.get("products", {})
    
    print(tool_use_id, tool["input"])
    if product_id in products:
        return {
            "toolUseId": tool_use_id,
            "status": "success",
            "content": [{"text": json.dumps(products[product_id])}]
        }

    agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "error",
        "content": [{"text": "Error: product not found"}]
    }
