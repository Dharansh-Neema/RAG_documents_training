from sentence_transformers import SentenceTransformer

import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
logger = setup_logger(name="embedder", log_file="logs/embedder.log")

BASE_DIR = Path(__file__).parent.parent
CHUNKS_DIR = BASE_DIR / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


def create_embeddings():
    """ 
    Create embeddings for the chunks and save them to a file.
    """
    logger.info("Creating embeddings for chunks")
    
    try:
        chunk_files = list(CHUNKS_DIR.glob("*.txt"))

        model = SentenceTransformer("all-MiniLM-L6-v2")
        for i,file in enumerate(chunk_files):
            with open(file, "r", encoding='utf-8') as f:
                text = f.read()
                embedding = model.encode(text)
                output_file = EMBEDDINGS_DIR / f"embedding_{i+1}.csv"
                pd.DataFrame(embedding).to_csv(output_file, index=False)
                logger.info(f"Successfully created embedding for {file.name}")
                    
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        return 0

if __name__ == "__main__":
    create_embeddings()