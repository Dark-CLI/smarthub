def update_summary(prev: str, user: str, assistant: str) -> str:
    """
    Super-simple rolling summary placeholder.
    Replace with a compact LLM-based summarizer if needed.
    """
    keep = (prev or "")[-600:]
    return (keep + f"\nU:{user}\nA:{assistant}")[-800:]
