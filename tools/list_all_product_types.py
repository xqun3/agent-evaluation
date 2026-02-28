from typing import Any
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "list_all_product_types",
    "description": "List the name and product id of all product types. Each product type has a variety of different items with unique item ids and options. There are only 50 product types in the store.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

# Function name must match tool name
def list_all_product_types(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    
    datas = agent.state.get("datas") or {}
    products = datas.get("products", {})
    
    product_dict = {
        product["name"]: product["product_id"] for product in products.values()
    }
    product_dict = dict(sorted(product_dict.items()))
    
    agent.state.set("datas", datas)

    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(product_dict)}]
    }
