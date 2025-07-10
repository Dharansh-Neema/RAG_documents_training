from utils import get_collection
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
logger = setup_logger(name="retriever", log_file="logs/retriever.log")

BASE_DIR = Path(__file__).parent.parent
VECTORDB_DIR = BASE_DIR / "vectordb"

def retriever(query):
    """
    To retrieve the documents from the vector database.
    """
    collection = get_collection()
    logger.info(f"Retrieving documents for query: {query}")
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    logger.info(f"Retrieved top 3 documents for query: {results}")
    return results

if __name__ == "__main__":
    print(retriever("What is the maximum number of items that can be listed on eBay?"))

    