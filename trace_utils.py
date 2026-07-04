"""
Helpers for extracting human-readable content from agent traces.
"""


def extract_final_response(trace: dict) -> str:
    """Return the last text block the agent emitted to the user.

    Walks the message list in reverse, finds the last assistant turn,
    and returns the concatenated text blocks. Returns a sentinel string
    if the agent never emitted prose (e.g. stopped mid-tool-loop).
    """
    messages = trace.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        texts = []
        for block in content:
            if hasattr(block, "type"):
                if block.type == "text":
                    texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        if texts:
            return " ".join(texts).strip()
    return "[NO FINAL RESPONSE]"
