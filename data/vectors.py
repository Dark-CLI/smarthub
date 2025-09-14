from __future__ import annotations
from typing import Iterable, List, Tuple, Optional
import lancedb
from lancedb.pydantic import LanceModel, Vector

_DB_PATH = "./lancedb"
_TBL_NAME = "main"

_db = None
_tbl = None
_DIM: Optional[int] = None


def _connect():
    global _db
    if _db is None:
        _db = lancedb.connect(_DB_PATH)
    return _db


def _ensure_table(dim: int):
    """Create or open LanceDB table, enforcing a fixed vector dimension."""
    global _tbl, _DIM
    db = _connect()
    if _tbl is None:
        if _TBL_NAME in db.table_names():
            _tbl = db.open_table(_TBL_NAME)
            # infer dim from existing schema
            schema = _tbl.schema
            _DIM = None
            for f in schema:
                if f.name == "vector":
                    # FixedSizeList
                    try:
                        _DIM = f.type.list_size
                    except Exception:
                        _DIM = dim
                    break
            if _DIM is None:
                _DIM = dim
        else:
            # define schema class with fixed dimension
            class Row(LanceModel):
                key: str
                vector: Vector(dim)
            _tbl = db.create_table(_TBL_NAME, schema=Row)
            _DIM = dim
    if _DIM != dim:
        raise ValueError(f"Vector dim {dim} != table dim {_DIM}. Call reset() to rebuild.")


def _k(key: Tuple[str, str] | str) -> str:
    return f"{key[0]}:{key[1]}" if isinstance(key, tuple) else str(key)


def add_or_update(key: Tuple[str, str] | str, vector: Iterable[float]) -> int:
    """Upsert by key (delete then add). Returns dummy row id."""
    vec = list(map(float, vector))
    _ensure_table(len(vec))
    kid = _k(key)
    _tbl.delete(f"key == '{kid}'")
    _tbl.add([{"key": kid, "vector": vec}])
    return 0


def add_or_update_batch(pairs: List[Tuple[Tuple[str, str] | str, Iterable[float]]]) -> List[int]:
    if not pairs:
        return []
    dim = len(list(pairs[0][1]))
    _ensure_table(dim)
    keys = [_k(k) for k, _ in pairs]
    if keys:
        _tbl.delete(" or ".join([f"key == '{kid}'" for kid in keys]))
    rows = [{"key": _k(k), "vector": list(map(float, v))} for k, v in pairs]
    _tbl.add(rows)
    return [0] * len(rows)


def query(vector: Iterable[float], top_k: int = 5) -> List[Tuple[str, float]]:
    vec = list(map(float, vector))
    _ensure_table(len(vec))
    res = _tbl.search(vec).metric("cosine").limit(top_k).to_list()
    # LanceDB returns either 'score' (similarity) or '_distance' (cosine distance)
    out: List[Tuple[str, float]] = []
    for r in res:
        score = r.get("score")
        if score is None:
            dist = r.get("_distance")
            score = 1.0 - float(dist) if dist is not None else 0.0
        out.append((r["key"], float(score)))
    return out


def reset():
    """Drop table so next call recreates it (use when changing embed dimension)."""
    db = _connect()
    if _TBL_NAME in db.table_names():
        db.drop_table(_TBL_NAME)
    global _tbl, _DIM
    _tbl, _DIM = None, None


def count() -> int:
    if _tbl is None:
        return 0
    return _tbl.count_rows()
