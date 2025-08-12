from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory

stores: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str):
    if session_id not in stores:
        stores[session_id] = InMemoryChatMessageHistory()
    return stores[session_id]