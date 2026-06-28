from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

def split_text_into_chunks(text: str, chunk_size: int = 450, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into smaller chunks using RecursiveCharacterTextSplitter.
    It attempts to split on logical boundaries such as paragraphs and sentences.
    
    Args:
        text (str): The raw document text content.
        chunk_size (int): Max character length of each chunk.
        chunk_overlap (int): Number of characters to overlap between contiguous chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    
    return text_splitter.split_text(text)
