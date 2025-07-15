from hashlib import sha256
from typing import Any, Callable, Dict, List, Type, Optional, Set, Union, Tuple

ToHashable = Union[
    str, int, float, Dict[str, "ToHashable"], List["ToHashable"], Set["ToHashable"]
]
Hashable = Union[str, int, float, Tuple["Hashable"], Tuple[Tuple[str, "Hashable"]]]

def to_hashable(item: ToHashable) -> Hashable:
    if isinstance(item, dict):
        return tuple((key, to_hashable(value)) for key, value in sorted(item.items()))
    elif isinstance(item, list):
        return tuple(to_hashable(element) for element in item)
    elif isinstance(item, set):
        return tuple(sorted(to_hashable(element) for element in item))
    else:
        return item

def consistent_hash(
    value: Hashable,
) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()


def generate_conversation(client, model_id, messages, system_prompt=None,max_token=8192) -> str:
    # request = json.dumps(native_request)
    try:
        inference_config = {
            "maxTokens": max_token,
            # "temperature": 0.9,
            # "topP":0.5,
        }
        
        parameters = {
            "modelId":model_id,
            "messages": messages,
            "system": [{"text":system_prompt}],
            "inferenceConfig": inference_config
            }
        if not system_prompt:
            parameters.pop("system") 

        response = client.converse_stream(
            **parameters
        )

        stream = response.get('stream')
        out = []
        for event in stream:
            if 'messageStart' in event:
                print(f"\nRole: {event['messageStart']['role']}")

            if 'contentBlockDelta' in event:
                print(event['contentBlockDelta']['delta']['text'], end="")
                out.append(event['contentBlockDelta']['delta']['text'])

            if 'messageStop' in event:
                print(f"\nStop reason: {event['messageStop']['stopReason']}")

            if 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata:
                    print("\nToken usage")
                    print(f"Input tokens: {metadata['usage']['inputTokens']}")
                    print(
                        f":Output tokens: {metadata['usage']['outputTokens']}")
                    print(f":Total tokens: {metadata['usage']['totalTokens']}")
                if 'metrics' in event['metadata']:
                    print(
                        f"Latency: {metadata['metrics']['latencyMs']} milliseconds")
        return "".join(out)
    except Exception as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        # time.sleep(1)
        return str(e)

def get_data_hash(data) -> str:
    return consistent_hash(to_hashable(data))