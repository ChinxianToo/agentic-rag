import os
import tiktoken

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def estimate_context_tokens(messages: list) -> int:
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")

    total = 0
    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            total += len(encoding.encode(str(msg.content)))
    return total
