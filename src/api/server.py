"""
FastAPI backend for CiteCare.

Run:
    uvicorn src.api.server:app --reload --port 8000
"""

import logging
import time
import uuid
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import COLLECTIONS_DIR, DEFAULT_RETENTION_DAYS
from src.document_loader import load_and_split
from src.vectorstore import (
    list_collections,
    delete_collection,
    load_collection,
    build_or_update_collection_from_dir,
    purge_expired_collections,
    get_collection_stats,
)
from src.rag_chain import create_rag_chain_with_sources
from src.utils.validation import validate_collection_name

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s","request_id":"%(request_id)s"}',
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CiteCare API", version="0.1.0")

# CORS (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Simple API key auth + tenant handling
API_KEY = os.getenv("CITECARE_API_KEY")


def get_auth_tenant(x_api_key: Optional[str] = Header(None), x_tenant: Optional[str] = Header("public")):
    if API_KEY:
        if x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    tenant = x_tenant or "public"
    valid, msg = validate_collection_name(tenant if len(tenant) >= 3 else f"{tenant}__")  # len check
    if not valid:
        raise HTTPException(status_code=400, detail=f"Invalid tenant: {msg}")
    return tenant


# Middleware to add request_id and measure latency
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.time()

    response = None
    try:
        response = await call_next(request)
    finally:
        duration = (time.time() - start) * 1000
        logger.info(
            "request complete",
            extra={
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "status": response.status_code if response else "error",
                "duration_ms": round(duration, 2),
            },
        )
    response.headers["X-Request-ID"] = request_id
    return response


class CollectionCreate(BaseModel):
    name: str
    incremental: bool = True
    retention_days: int = DEFAULT_RETENTION_DAYS


class QueryRequest(BaseModel):
    collection: str
    question: str
    k: Optional[int] = None


class RetentionUpdate(BaseModel):
    retention_days: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    # Simple readiness check: embeddings initialization
    from src.embeddings import get_embeddings

    try:
        _ = get_embeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embeddings unavailable: {e}")
    return {"status": "ready"}


@app.get("/collections")
def list_all_collections(tenant: str = Depends(get_auth_tenant)):
    cols = list_collections(tenant=tenant)
    return {"collections": cols, "tenant": tenant}


@app.post("/collections")
def create_collection_api(body: CollectionCreate, tenant: str = Depends(get_auth_tenant)):
    valid, msg = validate_collection_name(body.name)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    coll_dir = Path(COLLECTIONS_DIR) / tenant / body.name
    if not coll_dir.exists():
        raise HTTPException(status_code=400, detail="No documents found for this collection. Upload via UI first.")

    chunks = load_and_split(str(coll_dir), collection_name=body.name)
    vectorstore = build_or_update_collection_from_dir(
        collection_name=body.name,
        directory=coll_dir,
        documents=chunks,
        incremental=body.incremental,
        tenant=tenant,
        retention_days=body.retention_days,
    )
    stats = vectorstore._collection.count()
    return {"name": body.name, "chunks": stats, "incremental": body.incremental, "tenant": tenant}


@app.delete("/collections/{name}")
def delete_collection_api(name: str, tenant: str = Depends(get_auth_tenant)):
    deleted = delete_collection(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"deleted": name}


@app.post("/query")
def query(body: QueryRequest, tenant: str = Depends(get_auth_tenant)):
    valid, msg = validate_collection_name(body.collection)
    if not valid:
        raise HTTPException(status_code=400, detail=msg)

    vectorstore = load_collection(body.collection, tenant=tenant)
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    k_val = body.k or 5
    rag_func = create_rag_chain_with_sources(vectorstore, k=k_val)
    result = rag_func(body.question)
    return result


@app.patch("/collections/{name}/retention")
def update_retention(name: str, body: RetentionUpdate, tenant: str = Depends(get_auth_tenant)):
    coll_dir = Path(COLLECTIONS_DIR) / tenant / name
    meta_path = coll_dir / ".meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Collection not found")
    meta = {}
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {}
    meta["retention_days"] = body.retention_days
    meta["updated_at"] = datetime.utcnow().isoformat()
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"name": name, "tenant": tenant, "retention_days": body.retention_days}


@app.post("/collections/purge-expired")
def purge_expired(tenant: str = Depends(get_auth_tenant)):
    deleted = purge_expired_collections()
    return {"deleted": deleted}


@app.get("/collections/{name}/stats")
def collection_stats(name: str, tenant: str = Depends(get_auth_tenant)):
    stats = get_collection_stats(name, tenant=tenant)
    if not stats:
        raise HTTPException(status_code=404, detail="Collection not found")
    return {"tenant": tenant, **stats}
