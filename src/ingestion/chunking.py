import logging
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


# Configure logging for production-style output
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ingest_policy(pdf_path: str):
    """
        Converts a PDF to Markdown and chunks it by structural headers.
    """
    logger.info(f"Starting ingestion for policy document: {pdf_path}")
    
    try:
        # Convert PDF directly to Markdown to preserve headings
        md_text = pymupdf4llm.to_markdown(pdf_path)
        logger.info("Successfully converted PDF to Markdown.")
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise
    
    # Define Markdown headers to split on
    headers_to_split_on = [
        ("#", "Section"),
        ("##", "Subsection"),
        ("###", "Sub-subsection"),
    ]
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on, strip_headers = False)
    
    # Split the text into meaningful structural chunks
    chunks = markdown_splitter.split_text(md_text)
    logger.info(f"Generated {len(chunks)} raw chunks based on Markdown headers.")
    
    # Inject traceability metadata required by the assignment
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = f"chunk_{i}"
        chunk.metadata['source'] = pdf_path.split('/')[-1]
        
    logger.info("Successfully injected traceability metadata into all chunks.")
    return chunks


if __name__ == "__main__":
    target_pdf = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    chunks = ingest_policy(target_pdf)
    
    if chunks and len(chunks) > 10:
        logger.info(f"Sample Chunk [10] Metadata: {chunks[10].metadata}")
        logger.info(f"Sample Chunk [10] Content preview: {chunks[10].page_content[:250]}...")
