import uuid
def request_id() -> str:
    return str(uuid.uuid4())
