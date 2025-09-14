from sqlmodel import SQLModel, Field, Column, JSON
from typing import Any, Dict, Optional, List
from pydantic import ConfigDict

class Session(SQLModel, table=True):
    chat_id: str = Field(primary_key=True)
    tenant_id: Optional[str] = None
    summary_text: str = ""
    updated_at: int = 0

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_id: str
    role: str            # user|assistant|tool
    content: str         # text or compact JSON
    created_at: int = 0

class Device(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str
    name: str
    area: str
    domain: str
    type: str
    aliases: List[str] = Field(sa_column=Column(JSON))
    capabilities: List[str] = Field(sa_column=Column(JSON))
    hash: str

class Action(SQLModel, table=True):
    id: str = Field(primary_key=True)
    device_id: str
    name: str
    schema_hash: str

class Schema(SQLModel):
    schema_data: Dict[str, Any] = Field(default_factory=dict, alias="schema_json")
    hint_json: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(populate_by_name=True)
