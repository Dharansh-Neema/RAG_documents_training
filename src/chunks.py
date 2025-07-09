import sys
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
logger = setup_logger(name="chunks", log_file="logs/chunks.log")

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
CHUNKS_DIR = BASE_DIR / "chunks"

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def chunk_document(file_path, output_dir, chunk_size=250, chunk_overlap=40):
    """
    Split a document into chunks of approximately 250 words using sentence-aware splitting.
    """
    logger.info(f"Chunking document: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        document_chunks_dir = output_dir 
        document_chunks_dir.mkdir(parents=True, exist_ok=True)
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(text.split()),
            is_separator_regex=False
        )
        
        chunks = text_splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks from {file_path}")
        
        for i, chunk in enumerate(chunks):
            chunk_file = document_chunks_dir / f"chunk_{i+1:03d}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
            
            word_count = len(chunk.split())
            logger.debug(f"Chunk {i+1}: {word_count} words")
        
        logger.info(f"Successfully saved {len(chunks)} chunks")
        return len(chunks)
    
    except Exception as e:
        logger.error(f"Error chunking document {file_path}: {str(e)}")
        return 0


def process_training_document():
    """
    Process the Document: chunk it and save the chunks.
    """
    logger.info("Processing Document for chunking")
    
    try:
        document_path = PROCESSED_DATA_DIR / "training.txt"
        
        if not document_path.exists():
            logger.error(f"Document not found: {document_path}")
            return 0
        num_chunks = chunk_document(document_path, CHUNKS_DIR)
        
        logger.info(f"Completed chunking. Created {num_chunks} chunks")
        return num_chunks
    
    except Exception as e:
        logger.error(f"Error processing document for chunking: {str(e)}")
        return 0


if __name__ == "__main__":
    num_chunks = process_training_document()
    logger.info(f"Document chunking process completed. Created {num_chunks} chunks.")
