# smarthub/data/vectors.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Dict, Any
import time
from pathlib import Path
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Absolute DB path anchored at project root: smarthub/lancedb/
_DB_PATH = str((Path(__file__).resolve().parents[1] / "lancedb").as_posix())
_TBL_NAME = "main"

_db = None
_tbl = None
_DIM: Optional[int] = None


def _connect():
    global _db
    if _db is None:
        Path(_DB_PATH).mkdir(parents=True, exist_ok=True)
        _db = lancedb.connect(_DB_PATH)
    return _db


def _infer_dim_from_schema() -> Optional[int]:
    global _tbl
    if _tbl is None:
        return None
    schema = _tbl.schema
    for f in schema:
        if f.name == "vector":
            try:
                return f.type.list_size
            except Exception:
                return None
    return None


def _open_if_exists() -> bool:
    global _tbl, _DIM
    db = _connect()
    if _tbl is not None:
        return True
    if _TBL_NAME in db.table_names():
        _tbl = db.open_table(_TBL_NAME)
        _DIM = _infer_dim_from_schema()
        return True
    return False


def _ensure_table(dim: int):
    """
    Ensure table exists. If creating fresh, include metadata columns.
    If table already exists, rely on LanceDB schema evolution when adding rows.
    """
    global _tbl, _DIM
    db = _connect()
    if _tbl is None:
        if _TBL_NAME in db.table_names():
            _tbl = db.open_table(_TBL_NAME)
            existing_dim = _infer_dim_from_schema()
            _DIM = existing_dim if existing_dim is not None else dim
        else:
            class Row(LanceModel):
                key: str
                vector: Vector(dim)
                domain: Optional[str] = None
                area: Optional[str] = None
                content_hash: Optional[str] = None
                last_embedded_hash: Optional[str] = None
                updated_at: Optional[int] = None
            _tbl = db.create_table(_TBL_NAME, schema=Row)
            _DIM = dim
    if _DIM is not None and _DIM != dim:
        raise ValueError(f"Vector dim {dim} != table dim {_DIM}. Call reset() to rebuild.")


def _k(key: Tuple[str, str] | str) -> str:
    return f"{key[0]}:{key[1]}" if isinstance(key, tuple) else str(key)


def add_or_update_batch_with_meta(
    pairs: List[Tuple[Tuple[str, str] | str, Iterable[float], Dict[str, Any]]]
) -> None:
    """
    Upsert vectors with metadata. We DO NOT filter fields against current schema,
    allowing LanceDB to evolve and add new columns automatically if needed.
    """
    if not pairs:
        return
    vec0 = list(map(float, list(pairs[0][1])))
    _ensure_table(len(vec0))

    # delete any existing keys to simulate upsert
    keys = [_k(k) for (k, _, __) in pairs]
    if keys:
        _tbl.delete(" or ".join([f"key == '{kid}'" for kid in keys]))

    now = int(time.time())

    rows = []
    for k, v, meta in pairs:
        base: Dict[str, Any] = {
            "key": _k(k),
            "vector": list(map(float, v)),
            "updated_at": now,
        }
        # carry through all expected metadata (no filtering)
        for fld in ("domain", "area", "content_hash", "last_embedded_hash"):
            if fld in meta:
                base[fld] = meta[fld]
        rows.append(base)

    _tbl.add(rows)


def query(vector: Iterable[float], top_k: int = 5) -> List[Tuple[str, float]]:
    vec = list(map(float, vector))
    if not _open_if_exists():
        return []
    global _DIM
    if _DIM is None:
        _DIM = _infer_dim_from_schema()
    res = _tbl.search(vec).metric("cosine").limit(top_k).to_list()
    out: List[Tuple[str, float]] = []
    for r in res:
        score = r.get("score")
        if score is None:
            dist = r.get("_distance")
            score = 1.0 - float(dist) if dist is not None else 0.0
        out.append((r["key"], float(score)))
    return out


def get_meta(key: Tuple[str, str] | str) -> Optional[Dict[str, Any]]:
    if not _open_if_exists():
        return None
    kid = _k(key)
    rows = list(_tbl.to_lance().to_table().to_pylist())
    for r in rows:
        if r.get("key") == kid:
            return r
    return None


def get_last_hash(key: Tuple[str, str] | str) -> Optional[str]:
    meta = get_meta(key)
    return None if not meta else meta.get("last_embedded_hash")


def reset():
    db = _connect()
    if _TBL_NAME in db.table_names():
        db.drop_table(_TBL_NAME)
    global _tbl, _DIM
    _tbl, _DIM = None, None


def count() -> int:
    if not _open_if_exists():
        return 0
    return _tbl.count_rows()
