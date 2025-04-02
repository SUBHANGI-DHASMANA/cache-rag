# Vector search and indexing
import numpy as np
from usearch.index import Index
from typing import List, Tuple, Optional
import streamlit as st
from sentence_transformers import SentenceTransformer
import torch
from .config import logger

def build_search_index(chunks: List[str], embedding_model: SentenceTransformer) -> Tuple[Optional[Index], Optional[np.ndarray]]:
    if not chunks:
        raise ValueError("No chunks provided to build index")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Generating embeddings...")
        
        dim = embedding_model.get_sentence_embedding_dimension()
        index = Index(ndim=dim, metric='cos')
        batch_size = 32
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        all_embeddings = []
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            progress = (i + 1) / num_batches
            progress_bar.progress(progress)
            status_text.text(f"Generating embeddings (batch {i+1}/{num_batches})...")
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            batch_embeddings = embedding_model.encode(
                batch_chunks,
                show_progress_bar=False,
                device=str(embedding_model.device),
                convert_to_numpy=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            if i % 5 == 0 and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
        embeddings = np.vstack(all_embeddings)
        status_text.text("Building search index...")
        index.add(np.arange(len(chunks)), embeddings)
        
        progress_bar.empty()
        status_text.empty()
        
        return index, embeddings
        
    except Exception as e:
        logger.error(f"Error building search index: {e}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        raise RuntimeError(f"Failed to build search index: {str(e)}")

def get_relevant_chunks(
    query: str, 
    embedding_model: SentenceTransformer,
    index: Index,
    chunks: List[str],
    rag_cache: dict,
    top_k: int = 5
) -> List[str]:
    if not query.strip():
        return []
    
    try:
        cache_key = f"query_result_{hash(query)}_{top_k}"
        if cache_key in rag_cache:
            return rag_cache[cache_key]
            
        query_embedding = embedding_model.encode(
            query,
            device=str(embedding_model.device),
            convert_to_numpy=True
        )
        
        try:
            matches = index.search(np.array(query_embedding), top_k)
            result_chunks = [chunks[i] for i in matches.keys if i < len(chunks)]
        except Exception as search_error:
            logger.error(f"Error in vector search: {search_error}")
            result_chunks = []
            query_terms = set(query.lower().split())
            chunk_scores = []
            
            for i, chunk in enumerate(chunks):
                chunk_lower = chunk.lower()
                score = sum(1 for term in query_terms if term in chunk_lower)
                chunk_scores.append((score, i))
                
            chunk_scores.sort(reverse=True)
            result_chunks = [chunks[i] for _, i in chunk_scores[:top_k]]
            
        rag_cache[cache_key] = result_chunks
        return result_chunks
        
    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        return []