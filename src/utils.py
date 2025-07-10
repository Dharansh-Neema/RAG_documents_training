
from chromadb.config import Settings
import chromadb
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
from sentence_transformers import SentenceTransformer
logger = setup_logger(name="utils", log_file="logs/utils.log")
BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "chunks"
VECTORDB_DIR = BASE_DIR / "vectordb"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"


def get_or_create_collection():
    """
    To get or create the collection for chromadb.
    """
    logger.info("Getting or creating collection for chromadb")
    try:
        persist_dir = str(VECTORDB_DIR.resolve())
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        print(chroma_client.heartbeat())


        collection = chroma_client.create_collection(
            name="ebay",
        )

        logger.info("Collection setup successfully")
        return collection

    except Exception as e:
        logger.error(f"Error setting up collection: {str(e)}")
        return None

def get_collection():
    """
    To get the collection for chromadb.
    """
    logger.info("Getting collection for chromadb")
    try:
        persist_dir = str(VECTORDB_DIR.resolve())
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        collection = chroma_client.get_collection(name="ebay")
        logger.info("Collection fetched successfully")
        return collection
    except Exception as e:
        logger.error(f"Error fetching collection: {str(e)}")
        return None
if __name__ == "__main__":
    collection = get_or_create_collection()
    print(collection)