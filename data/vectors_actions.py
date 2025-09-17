# data/vectors_actions.py
from typing import List, Tuple, Dict, Any
import os, hashlib
import lancedb

_DB_PATH = os.getenv("LANCEDB_PATH", "./.lancedb")
_TABLE = "actions_index"

def _db():
    return lancedb.connect(_DB_PATH)

def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def add_or_update(rows: List[Dict[str, Any]]):
    """
    rows: [{"key":"service:<domain.service>", "vector":[...], "snapshot":"<json>"}]
    Upsert via delete-then-add on key.
    """
    if not rows:
        return
    db = _db()
    data = [{"key": r["key"], "vector": r["vector"], "last_embedding_hash": _hash(r["snapshot"])} for r in rows]
    if _TABLE not in db.table_names():
        db.create_table(_TABLE, data=data)
        return
    tbl = db.open_table(_TABLE)
    keys = [d["key"] for d in data]
    quoted = ",".join([f"'{k}'" for k in keys])
    tbl.delete(f"key IN ({quoted})")
    tbl.add(data)

def reset():
    db = _db()
    if _TABLE in db.table_names():
        db.drop_table(_TABLE)

def query(qvec: List[float], top_k: int = 6) -> List[Tuple[str, float]]:
    db = _db()
    tbl = db.open_table(_TABLE)
    res = tbl.search(qvec).limit(top_k).to_list()
    out: List[Tuple[str, float]] = []
    for r in res:
        key = r.get("key")
        if "_distance" in r:
            dist = float(r["_distance"])
            score = 1.0 - dist
        else:
            score = float(r.get("score", 0.0))
        out.append((key, score))
    return out
