META_PATTERNS = (
    "what did i say", "what did i just say", "what was my last message",
    "previous message", "last message", "recap", "summarize our chat",
    "what did you say", "you said earlier", "remind me what we discussed",
    "what was your previous response", "what did we talk about"
)

def is_meta_request(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in META_PATTERNS)