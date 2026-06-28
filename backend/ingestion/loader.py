import os
import uuid
import chromadb
from typing import Dict, List, Any
from backend import config
from backend.services.embedder import embedder
from backend.ingestion.chunker import split_text_into_chunks

def get_chroma_client() -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client pointing to the configured folder."""
    return chromadb.PersistentClient(path=config.CHROMA_STORE_DIR)

def get_collections(client: chromadb.PersistentClient) -> Dict[str, chromadb.Collection]:
    """
    Retrieves or creates the three core ChromaDB collections.
    Crucial: Metadata 'hnsw:space' is set to 'cosine' to ensure cosine distance is used.
    """
    return {
        "drugs": client.get_or_create_collection(
            name="drugs", 
            metadata={"hnsw:space": "cosine"}
        ),
        "diseases": client.get_or_create_collection(
            name="diseases", 
            metadata={"hnsw:space": "cosine"}
        ),
        "guidelines": client.get_or_create_collection(
            name="guidelines", 
            metadata={"hnsw:space": "cosine"}
        ),
    }

def extract_drug_metadata(text: str) -> Dict[str, str]:
    """Helper to parse drug name and category from text for richer search metadata."""
    metadata = {}
    lower_text = text.lower()
    
    # Simple rule-based extraction for sample drug data
    if "ibuprofen" in lower_text:
        metadata["drug_name"] = "ibuprofen"
        metadata["category"] = "NSAID"
    elif "metformin" in lower_text:
        metadata["drug_name"] = "metformin"
        metadata["category"] = "Biguanide"
    elif "warfarin" in lower_text:
        metadata["drug_name"] = "warfarin"
        metadata["category"] = "Anticoagulant"
    elif "aspirin" in lower_text:
        metadata["drug_name"] = "aspirin"
        metadata["category"] = "Salicylate"
        
    return metadata

def populate_database(overwrite: bool = False):
    """
    Scans the knowledge_base folder, chunks files, embeds them,
    and inserts them into ChromaDB if collections are empty.
    """
    client = get_chroma_client()
    collections = get_collections(client)
    
    # We map collection folders to their Chroma collection objects
    folder_mapping = {
        "drugs": collections["drugs"],
        "diseases": collections["diseases"],
        "guidelines": collections["guidelines"]
    }
    
    for folder_name, collection in folder_mapping.items():
        # Check current count in ChromaDB
        count = collection.count()
        if count > 0 and not overwrite:
            print(f"[LOADER] Collection '{folder_name}' already populated ({count} documents). Skipping.")
            continue
            
        if overwrite and count > 0:
            print(f"[LOADER] Overwriting '{folder_name}' collection. Deleting existing vectors...")
            client.delete_collection(folder_name)
            # Re-fetch it
            collection = client.get_or_create_collection(
                name=folder_name, 
                metadata={"hnsw:space": "cosine"}
            )
            folder_mapping[folder_name] = collection
            
        target_dir = os.path.join(config.KNOWLEDGE_BASE_DIR, folder_name)
        if not os.path.exists(target_dir):
            print(f"[LOADER] Warning: Directory {target_dir} does not exist. Creating it.")
            os.makedirs(target_dir, exist_ok=True)
            continue
            
        # Scan files in directory
        files = [f for f in os.listdir(target_dir) if f.endswith(('.txt', '.md'))]
        if not files:
            print(f"[LOADER] No text or markdown files found in {target_dir}.")
            continue
            
        for file_name in files:
            file_path = os.path.join(target_dir, file_name)
            print(f"[LOADER] Processing: {file_path}")
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Chunk the file contents
            chunks = split_text_into_chunks(content)
            if not chunks:
                continue
                
            print(f"[LOADER] Splitting into {len(chunks)} chunks.")
            
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{folder_name[:-1] if folder_name.endswith('s') else folder_name}_{uuid.uuid4()}"
                
                # Setup core chunk metadata
                meta = {
                    "source_file": file_name,
                    "collection": folder_name,
                    "chunk_index": i,
                    "chunk_size": len(chunk)
                }
                
                # Add domain-specific tags if drugs
                if folder_name == "drugs":
                    meta.update(extract_drug_metadata(chunk))
                    
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(meta)
                
            # Compute embeddings in a single batch to optimize CPU
            print(f"[LOADER] Generating embeddings for {len(chunks)} chunks...")
            embeddings = embedder.embed_documents(documents)
            
            # Upsert into ChromaDB
            print(f"[LOADER] Writing vectors to ChromaDB...")
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"[LOADER] Ingested {len(chunks)} chunks from '{file_name}' into collection '{folder_name}'.")

if __name__ == "__main__":
    populate_database(overwrite=True)
