"""
Vector store operations with multiple collection support, incremental indexing,
retention metadata, versioning, and basic purge capability.
"""

from pathlib import Path
from typing import List, Optional
import shutil
import json
from datetime import datetime

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.config import CHROMA_PERSIST_DIR, RETRIEVAL_K, DEFAULT_RETENTION_DAYS
from src.embeddings import get_embeddings
from src.utils.checksums import compute_dir_checksums


def get_collection_path(collection_name: str, tenant: str = "public") -> str:
    """Get the path for a specific collection within a tenant."""
    return f"{CHROMA_PERSIST_DIR}/{tenant}/{collection_name}"


def list_collections(tenant: str = "public") -> List[str]:
    """List all available collections for a tenant."""
    base_path = Path(CHROMA_PERSIST_DIR) / tenant
    if not base_path.exists():
        return []

    collections = []
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if (item / "chroma.sqlite3").exists():
                collections.append(item.name)
    return sorted(collections)


def create_collection(
    collection_name: str,
    documents: List[Document],
    *,
    tenant: str = "public",
    retention_days: int = DEFAULT_RETENTION_DAYS,
    owner: Optional[str] = None,
    version: int = 1,
) -> Chroma:
    """
    Create a new collection with documents (rebuild).
    """
    persist_dir = Path(get_collection_path(collection_name, tenant))
    embeddings = get_embeddings()

    # Remove existing if present
    if persist_dir.exists():
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name=collection_name
    )

    meta_path = persist_dir / ".meta.json"
    meta = {
        "collection": collection_name,
        "tenant": tenant,
        "retention_days": retention_days,
        "owner": owner,
        "version": version,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    _save_meta(meta_path, meta)

    print(f"Created collection '{collection_name}' (tenant={tenant}) with {len(documents)} chunks")
    return vectorstore


def add_documents_to_collection(
    collection_name: str,
    documents: List[Document],
    *,
    tenant: str = "public"
) -> Optional[Chroma]:
    """
    Add documents to an existing collection (incremental).
    """
    vectorstore = load_collection(collection_name, tenant=tenant)
    if not vectorstore:
        return None
    if documents:
        vectorstore.add_documents(documents)
    return vectorstore


def load_collection(collection_name: str, *, tenant: str = "public") -> Optional[Chroma]:
    """
    Load an existing collection.
    """
    persist_dir = Path(get_collection_path(collection_name, tenant))

    if not persist_dir.exists():
        print(f"Collection '{collection_name}' not found")
        return None

    embeddings = get_embeddings()

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name=collection_name
    )

    count = vectorstore._collection.count()
    print(f"Loaded collection '{collection_name}' (tenant={tenant}) with {count} chunks")
    return vectorstore


def delete_collection(collection_name: str) -> bool:
    """Delete a collection."""
    persist_dir = Path(get_collection_path(collection_name))

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"Deleted collection '{collection_name}'")
        return True
    return False


def _load_meta(meta_path: Path) -> dict:
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_meta(meta_path: Path, meta: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2))


def get_collection_stats(collection_name: str, *, tenant: str = "public") -> dict:
    """Get statistics for a collection."""
    vectorstore = load_collection(collection_name, tenant=tenant)
    if not vectorstore:
        return {}

    count = vectorstore._collection.count()

    # Get unique files
    results = vectorstore._collection.get(include=["metadatas"])
    files = set()
    for meta in results.get("metadatas", []):
        if meta and "file_name" in meta:
            files.add(meta["file_name"])

    return {
        "name": collection_name,
        "chunks": count,
        "files": list(files),
        "file_count": len(files)
    }


def similarity_search(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_K
) -> List[Document]:
    """Search for similar documents."""
    return vectorstore.similarity_search(query, k=k)


def similarity_search_with_score(
    vectorstore: Chroma,
    query: str,
    k: int = RETRIEVAL_K
) -> List[tuple]:
    """Search with relevance scores."""
    return vectorstore.similarity_search_with_score(query, k=k)


def get_retriever(vectorstore: Chroma, k: int = RETRIEVAL_K):
    """Get a retriever for use in chains."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


# Legacy support
def get_or_create_vectorstore(documents: List[Document], collection_name: str = "default"):
    """Get or create a vectorstore (legacy compatibility)."""
    existing = load_collection(collection_name)
    if existing:
        return existing
    if documents:
        return create_collection(collection_name, documents)
    raise ValueError("No collection found and no documents provided")


def build_or_update_collection_from_dir(
    collection_name: str,
    directory: Path,
    documents: List[Document],
    incremental: bool = True,
    *,
    tenant: str = "public",
    retention_days: int = DEFAULT_RETENTION_DAYS,
    owner: Optional[str] = None,
) -> Chroma:
    """
    Build or update a collection from a directory with incremental indexing.

    - Computes checksums for files in the directory
    - Skips re-embedding unchanged files when incremental=True
    """
    persist_dir = Path(get_collection_path(collection_name, tenant))
    meta_path = persist_dir / ".meta.json"
    previous_meta = _load_meta(meta_path) if incremental else {}
    current_meta = compute_dir_checksums(directory)
    previous_version = previous_meta.get("version", 1)

    # Identify changed or new files
    changed_files = {
        rel for rel, checksum in current_meta.items()
        if previous_meta.get(rel) != checksum
    }

    if not incremental or not persist_dir.exists():
        # Full rebuild
        persist_dir.mkdir(parents=True, exist_ok=True)
        vectorstore = create_collection(
            collection_name,
            documents,
            tenant=tenant,
            retention_days=retention_days,
            owner=owner,
            version=previous_version + 1 if previous_meta else 1,
        )
    else:
        vectorstore = load_collection(collection_name, tenant=tenant)
        if vectorstore is None:
            vectorstore = create_collection(
                collection_name,
                documents,
                tenant=tenant,
                retention_days=retention_days,
                owner=owner,
                version=previous_version + 1,
            )
        else:
            # Filter documents to only changed files
            if changed_files:
                docs_to_add = [
                    doc for doc in documents
                    if doc.metadata.get("source") and
                    Path(doc.metadata["source"]).relative_to(directory).__str__() in changed_files
                ]
                if docs_to_add:
                    vectorstore.add_documents(docs_to_add)

    # Save meta
    meta_to_save = {
        **current_meta,
        "collection": collection_name,
        "tenant": tenant,
        "retention_days": retention_days,
        "owner": owner,
        "version": previous_version + 1 if changed_files or not previous_meta else previous_version,
        "updated_at": datetime.utcnow().isoformat(),
        "created_at": previous_meta.get("created_at", datetime.utcnow().isoformat()),
    }
    _save_meta(meta_path, meta_to_save)
    return vectorstore


def purge_expired_collections(now: Optional[datetime] = None) -> List[str]:
    """
    Delete collections whose retention_days have expired.
    """
    now = now or datetime.utcnow()
    base_path = Path(CHROMA_PERSIST_DIR)
    deleted: List[str] = []
    if not base_path.exists():
        return deleted

    for tenant_dir in base_path.iterdir():
        if not tenant_dir.is_dir():
            continue
        for coll_dir in tenant_dir.iterdir():
            meta_path = coll_dir / ".meta.json"
            meta = _load_meta(meta_path)
            retention = meta.get("retention_days", DEFAULT_RETENTION_DAYS)
            created_at = meta.get("created_at")
            if created_at:
                try:
                    created_dt = datetime.fromisoformat(created_at)
                    age_days = (now - created_dt).days
                    if age_days > retention:
                        shutil.rmtree(coll_dir)
                        deleted.append(f"{tenant_dir.name}/{coll_dir.name}")
                except Exception:
                    continue
    return deleted
