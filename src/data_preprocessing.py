import re
import sys
from pathlib import Path
import PyPDF2
from bs4 import BeautifulSoup
import html2text
sys.path.append(str(Path(__file__).parent.parent))
from logger import setup_logger
from dotenv import load_dotenv
load_dotenv()

logger = setup_logger(name="data_preprocessing", log_file="logs/data_preprocessing.log")

# Directory for data and processed data
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = BASE_DIR / "processed_data"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        logger.debug(f"Successfully extracted {len(text)} characters.")
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""


def clean_text(text):
    """
    Clean the extracted text.
    """
    logger.info("Cleaning extracted text")
    
    try:
        # Remove HTML tags
        if bool(BeautifulSoup(text, "html.parser").find()):
            soup = BeautifulSoup(text, "html.parser")
            text = soup.get_text()
        
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        text = h.handle(text)
        
        # Remove unnecessary characters and lines 
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        logger.debug(f"Text cleaned successfully, final length: {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error cleaning text: {str(e)}")
        return text


def process_pdf_file(pdf_path, output_dir):
    """
    Process a PDF file: extract text, clean it, and save to a text file.
    """
    try:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return False
        
        cleaned_text = clean_text(text)
        if not cleaned_text:
            logger.warning(f"Cleaning resulted in empty text for {pdf_path}")
            return False
        
        output_file = output_dir / f"{pdf_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
        
        logger.info(f"Successfully processed {pdf_path} and saved to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False


def process_all_pdfs():
    """
    Process PDF file in the data directory and save the cleaned text to the processed_data directory.
    """
    logger.info(f"Starting to process all PDFs in {DATA_DIR}")
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(DATA_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    successful_count = 0
    for pdf_file in pdf_files:
        logger.info(f"Processing file {pdf_file.name}")
        if process_pdf_file(pdf_file, PROCESSED_DATA_DIR):
            successful_count += 1
    
    logger.info(f"Completed processing. Successfully processed {successful_count}")
    return successful_count


if __name__ == "__main__":
    logger.info("PDF ingestion process started")
    num_processed = process_all_pdfs()
    logger.info(f"PDF ingestion process completed. Processed {num_processed} files.")
