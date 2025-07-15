import json
import os
from collections import OrderedDict
from typing import Any

FOLDER_PATH = os.path.dirname(__file__)


def load_data() -> dict[str, Any]:
    file_mapping = [
        ("orders", "orders.json"),
        ("products", "products.json"),
        ("users", "users.json")
    ]
    
    result = {}
    for key, filename in file_mapping:
        with open(os.path.join(FOLDER_PATH, filename)) as f:
            # 使用 object_pairs_hook=OrderedDict 保证JSON对象内部的键顺序
            result[key] = json.load(f, object_pairs_hook=OrderedDict)
    
    return result
