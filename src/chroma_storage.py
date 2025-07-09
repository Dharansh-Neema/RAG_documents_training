from chromadb.config import Settings
import pandas as pd
import sys
from pathlib import Path
from utils import get_or_create_collection
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
logger = setup_logger(name="chroma_storage", log_file="logs/chroma_storage.log")
BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "chunks"
VECTORDB_DIR = BASE_DIR / "vectordb"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


def store_in_vector_db():
    """
    To store the embeddings in chromadb.
    """
    logger.info("Storing embeddings in chromadb")
    try:
        collection = get_or_create_collection()
        if collection is None:
            logger.error("Failed to get collection")
            return 0
        chunk_files = list(CHUNKS_DIR.glob("*.txt"))
        embedding_file = list(EMBEDDINGS_DIR.glob("*.csv"))
        # Storing in Vector DB 
        logger.info("Storing embeddings in chromadb")
        for i, chunk_file in enumerate(chunk_files):
            with open(chunk_file, "r", encoding='utf-8') as f:
                text = f.read()
                embedding = pd.read_csv(embedding_file[i])
                collection.add(
                    documents=[text],
                    metadatas=[{"chunk": i+1}],
                    embeddings=[embedding.values[0].tolist()],
                    ids=[f"id_{i+1}"]
                )
        
        logger.info("Successfully stored embeddings in chromadb")
        return len(chunk_files)
    
    except Exception as e:
        logger.error(f"Error storing embeddings: {str(e)}")
        return 0

if __name__ == "__main__":
    store_in_vector_db()

