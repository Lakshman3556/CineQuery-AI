import os
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
from backend import config
from backend.services.embedder import embedder
from backend.ingestion.loader import get_chroma_client, get_collections, extract_drug_metadata

router = APIRouter()

class IngestRequest(BaseModel):
    collection: str = Field(..., description="ChromaDB collection: 'drugs', 'diseases', or 'guidelines'")
    text: str = Field(..., description="Medical text to ingest")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata fields")

@router.post("/ingest")
async def ingest_endpoint(request: IngestRequest):
    """
    POST /ingest endpoint for admins to manually add chunks to ChromaDB.
    """
    col_name = request.collection.strip().lower()
    text = request.text.strip()
    meta = request.metadata
    
    if col_name not in ["drugs", "diseases", "guidelines"]:
        raise HTTPException(status_code=400, detail="Collection must be 'drugs', 'diseases', or 'guidelines'.")
    if not text:
        raise HTTPException(status_code=400, detail="Text content cannot be empty.")
        
    try:
        client = get_chroma_client()
        collections = get_collections(client)
        collection = collections[col_name]
        
        # Setup metadata
        chunk_id = f"{col_name[:-1] if col_name.endswith('s') else col_name}_{uuid.uuid4()}"
        chunk_metadata = {
            "source_file": "admin_api",
            "collection": col_name,
            "chunk_index": 0,
            "chunk_size": len(text)
        }
        
        # Extract drug details if applicable
        if col_name == "drugs":
            chunk_metadata.update(extract_drug_metadata(text))
            
        # Update with user provided metadata
        chunk_metadata.update(meta)
        
        # Generate embedding
        embedding = embedder.embed_text(text)
        
        # Add to ChromaDB
        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[chunk_metadata]
        )
        
        print(f"[INGEST] Chunks successfully added to collection '{col_name}' under ID '{chunk_id}'")
        return {
            "status": "success",
            "id": chunk_id,
            "collection": col_name,
            "message": "Content successfully ingested and indexed."
        }
    except Exception as e:
        print(f"[INGEST] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to ingest content: {str(e)}")
