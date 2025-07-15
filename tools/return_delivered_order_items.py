from typing import Any, List
from strands.types.tools import ToolResult, ToolUse
import json
from strands import Agent
from data import load_data

TOOL_SPEC = {
    "name": "return_delivered_order_items",
    "description": "Return some items of a delivered order. The order status will be changed to 'return requested'. The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. The user will receive follow-up email for how and where to return the item.",
    "inputSchema": {
        "json": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id."
                },
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list."
                },
                "payment_method_id": {
                    "type": "string",
                    "description": "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. These can be looked up from the user or order details."
                }
            },
            "required": ["order_id", "item_ids", "payment_method_id"]
        }
    }
}

# Function name must match tool name
def return_delivered_order_items(tool: ToolUse,  agent: Agent, **kwargs: Any) -> ToolResult:
    tool_use_id = tool["toolUseId"]
    order_id = tool["input"]["order_id"]
    item_ids = tool["input"]["item_ids"]
    payment_method_id = tool["input"]["payment_method_id"]
    
    if "datas" in kwargs:
        datas = kwargs["datas"]
    else:   
        datas = agent.state.get("datas") or {} 
    orders = datas.get("orders", {})

    # Check if the order exists and is delivered
    if order_id not in orders:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: order not found"}]
        }
    
    order = orders[order_id]
    if order["status"] != "delivered":
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: non-delivered order cannot be returned"}]
        }

    # Check if the payment method exists and is either the original payment method or a gift card
    if payment_method_id not in datas["users"][order["user_id"]]["payment_methods"]:
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: payment method not found"}]
        }
    
    if ("gift_card" not in payment_method_id and 
        payment_method_id != order["payment_history"][0]["payment_method_id"]):
        return {
            "toolUseId": tool_use_id,
            "status": "error",
            "content": [{"text": "Error: payment method should be either the original payment method or a gift card"}]
        }

    # Check if the items to be returned exist (there could be duplicate items in either list)
    all_item_ids = [item["item_id"] for item in order["items"]]
    for item_id in item_ids:
        if item_ids.count(item_id) > all_item_ids.count(item_id):
            return {
                "toolUseId": tool_use_id,
                "status": "error",
                "content": [{"text": "Error: some item not found"}]
            }

    # Update the order status
    order["status"] = "return requested"
    order["return_items"] = sorted(item_ids)
    order["return_payment_method_id"] = payment_method_id
    
    if agent is not None:
        agent.state.set("datas", datas)


    return {
        "toolUseId": tool_use_id,
        "status": "success",
        "content": [{"text": json.dumps(order)}]
    }
