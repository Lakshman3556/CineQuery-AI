import os
import chromadb
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from backend import config
from backend.services.embedder import embedder
from backend.ingestion.loader import get_chroma_client, get_collections

class MedicalRetriever:
    """
    Handles multi-collection vector search from ChromaDB and local 
    Cross-Encoder re-ranking to deliver highly relevant context.
    """
    def __init__(self):
        # Initialize the persistent ChromaDB client
        self.client = get_chroma_client()
        self.collections = get_collections(self.client)
        
        # Load the lightweight Cross-Encoder model locally for semantic re-ranking.
        # This model scores the direct query-document pairs, providing high-precision relevance.
        # Size: ~80MB, runs efficiently on CPU.
        print("[RETRIEVER] Initializing local CrossEncoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2'...")
        self.re_ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("[RETRIEVER] CrossEncoder loaded successfully.")

    def retrieve(self, query: str) -> Tuple[List[Dict[str, Any]], float]:
        """
        Main retrieval pipeline:
        1. Embed the query.
        2. Query ChromaDB collections (drugs, diseases, guidelines) in parallel.
        3. Translate distances to cosine similarity.
        4. Filter to top 10 by cosine similarity.
        5. Re-rank down to top 3 using the local Cross-Encoder.
        6. Compute confidence as the average cosine similarity of the top 3 chunks.
        
        Returns:
            Tuple[List[Dict[str, Any]], float]: (List of top 3 chunks, confidence score)
        """
        print(f"[RETRIEVER] Processing query: '{query}'")
        
        # 1. Embed the query locally
        query_vector = embedder.embed_text(query)
        
        # 2. Query all 3 collections
        all_retrieved = []
        for col_name, collection in self.collections.items():
            try:
                # Retrieve top-5 candidates per collection
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=5,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Check if results are empty
                if not results or not results["documents"] or not results["documents"][0]:
                    continue
                    
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]
                
                for i in range(len(documents)):
                    # Cosine distance in ChromaDB is [0, 2], so similarity = 1.0 - distance.
                    # We cap similarity between 0.0 and 1.0.
                    dist = distances[i]
                    similarity = max(0.0, min(1.0, 1.0 - dist))
                    
                    all_retrieved.append({
                        "id": ids[i],
                        "text": documents[i],
                        "metadata": metadatas[i],
                        "similarity": similarity,
                        "collection": col_name
                    })
            except Exception as e:
                print(f"[RETRIEVER] Error querying collection '{col_name}': {e}")
                continue
                
        if not all_retrieved:
            print("[RETRIEVER] No documents retrieved from any collection.")
            return [], 0.0
            
        # 4. Sort all results by original cosine similarity and select top-10
        all_retrieved.sort(key=lambda x: x["similarity"], reverse=True)
        top_10 = all_retrieved[:10]
        print(f"[RETRIEVER] Merged collections. Selected top-{len(top_10)} candidates by cosine similarity.")
        
        # 5. Re-rank top-10 down to top-3 using the Cross-Encoder
        pairs = [(query, chunk["text"]) for chunk in top_10]
        print(f"[RETRIEVER] Re-ranking with CrossEncoder...")
        cross_scores = self.re_ranker.predict(pairs)
        
        # Attach Cross-Encoder scores to the chunks
        for idx, score in enumerate(cross_scores):
            top_10[idx]["rerank_score"] = float(score)
            
        # Sort by the re-ranker score (highest first)
        top_10.sort(key=lambda x: x["rerank_score"], reverse=True)
        top_3 = top_10[:3]
        
        # 6. Calculate confidence score
        # Confidence score is the average cosine similarity of the top-3 selected chunks.
        if top_3:
            avg_similarity = sum(chunk["similarity"] for chunk in top_3) / len(top_3)
        else:
            avg_similarity = 0.0
            
        print(f"[RETRIEVER] Selected top-3 chunks. Avg Cosine Similarity (Confidence): {avg_similarity:.4f}")
        for i, chunk in enumerate(top_3):
            print(f"  Chunk {i+1} | Source: {chunk['metadata'].get('source_file')} | Cosine Sim: {chunk['similarity']:.4f} | Rerank Score: {chunk['rerank_score']:.4f}")
            
        return top_3, avg_similarity

# Export a single global instance for retrieval operations
retriever = MedicalRetriever()
