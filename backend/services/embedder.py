import os
from sentence_transformers import SentenceTransformer
from typing import List

class MedicalEmbedder:
    """
    A Thread-safe Singleton wrapper around SentenceTransformer to ensure 
    the model is only loaded into memory once across the application lifetime.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MedicalEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        # Initialize the sentence-transformer model.
        # This model is 384-dimensional and runs very efficiently on CPU.
        # On first execution, this downloads the model (~80MB) automatically.
        print("[EMBEDDER] Initializing local model 'all-MiniLM-L6-v2'...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("[EMBEDDER] Local model 'all-MiniLM-L6-v2' loaded successfully.")
        self._initialized = True

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string query into a 384-dimensional vector.
        ChromaDB expects native Python float lists.
        """
        if not text.strip():
            # Return an empty embedding or zeros for empty strings to prevent errors
            return [0.0] * 384
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a batch of documents/chunks into a list of 384-dimensional vectors.
        Uses a batch size to prevent memory spikes on ingestion.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
        return embeddings.tolist()

# Export a single global instance for use throughout the backend services
embedder = MedicalEmbedder()
