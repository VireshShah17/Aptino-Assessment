import os
import logging
import numpy as np
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest


# Load environment variables from .env file for secure deployment
load_dotenv()

# Configure logging for production-style output
logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy INFO logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)


class HybridRetriever:
    def __init__(self, chunks: List[Document]):
        """
            Initializes the Hybrid Retriever with dense (FAISS via Gemini), sparse (BM25) indices, 
            and an ultra-lightweight ONNX-based Reranker.
        """
        logger.info("Initializing Hybrid Retriever...")
        self.chunks = chunks

        # Securely fetch API key from the environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY environment variable is missing. Please set it in your .env file.")
            raise ValueError("Missing GOOGLE_API_KEY")

        # Initialize Dense Retriever (FAISS with Gemini Embeddings)
        logger.info("Loading Gemini embedding model for dense retrieval...")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = "models/gemini-embedding-001",
            google_api_key = api_key,
        )

        try:
            self.vector_store = FAISS.from_documents(self.chunks, self.embeddings)
        except Exception:
            logger.exception("Failed to build FAISS index from Gemini embeddings.")
            raise

        # Initialize Sparse Retriever (BM25)
        logger.info("Building BM25 index for sparse retrieval...")
        tokenized_corpus = [chunk.page_content.lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize Lightweight Reranker
        logger.info("Loading lightweight FlashRank reranker...")
        self.ranker = Ranker()
        
        logger.info("Hybrid Retriever initialized successfully.")


    def dense_search(self, query: str, k: int = 5) -> List[Document]:
        """
            Performs semantic search using FAISS.
        """
        docs = self.vector_store.similarity_search(query, k = k)
        return docs


    def sparse_search(self, query: str, k: int = 5) -> List[Document]:
        """
            Performs lexical search using BM25.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:k]

        docs = [self.chunks[i] for i in top_n_indices if scores[i] > 0]
        return docs


    def get_fused_results(self, query: str, k: int = 5) -> List[Document]:
        """
            Fuses results from dense and sparse retrievers, eliminating duplicates,
            and applies a reranking stage to re-order the evidence.
        """
        logger.info(f"Executing hybrid search for query: '{query}'")

        # Retrieve a broader pool of candidates before reranking
        pool_size = k * 2
        dense_results = self.dense_search(query, k = pool_size)
        sparse_results = self.sparse_search(query, k = pool_size)

        # Combine and deduplicate based on chunk_id
        fused_docs = {}

        for doc in dense_results + sparse_results:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id not in fused_docs:
                fused_docs[chunk_id] = doc

        unique_docs = list(fused_docs.values())
        logger.info(f"Retrieved {len(unique_docs)} unique documents before reranking.")

        # Prepare payload for FlashRank
        passages = [
            {"id": doc.metadata.get("chunk_id", str(i)), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(unique_docs)
        ]
        
        rerank_request = RerankRequest(query = query, passages = passages)
        reranked_results = self.ranker.rerank(rerank_request)
        
        # Reconstruct Document objects from reranked output
        final_docs = []
        for result in reranked_results[:k]:
            final_docs.append(Document(page_content = result["text"], metadata = result["meta"]))

        logger.info(f"Successfully reranked and returned top {k} documents.")
        return final_docs


if __name__ == "__main__":
    from src.ingestion.chunking import ingest_policy

    target_pdf = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    chunks = ingest_policy(target_pdf)

    retriever = HybridRetriever(chunks = chunks)

    test_query = "What is the waiting period for pre-existing diseases?"
    results = retriever.get_fused_results(query = test_query, k = 3)

    for idx, res in enumerate(results):
        logger.info(f"Result {idx + 1} Metadata: {res.metadata}")
        logger.info(f"Result {idx + 1} Snippet: {res.page_content[:150]}...\n")
