from langchain_core.messages.utils import trim_messages

def last_k(history, k=6):
    return trim_messages(
        history,
        token_counter=len,
        max_tokens=k,
        strategy="last",
        start_on="human",
        include_system=True,
    )