# Main Streamlit application
import streamlit as st
import time
from typing import Tuple, Optional, Dict, Any
import torch
import logging

# Import from our modules
from rag_app.config import MAX_FILE_SIZE_MB
from rag_app.models import load_embedding_model, load_llm_model
from rag_app.document_processing import extract_text_from_pdf
from rag_app.chunking import load_text_splitter, create_chunks
from rag_app.search import build_search_index, get_relevant_chunks
from rag_app.generation import generate_response
from rag_app.utils import setup_device, clear_gpu_memory
from rag_app.interface import initialize_session_state, show_sidebar, show_document_stats, show_main_interface

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize
device = setup_device()
initialize_session_state()
embedding_model = load_embedding_model(device)
llm_tokenizer, llm_model = load_llm_model(device) if embedding_model is not None else (None, None)
text_splitter = load_text_splitter()

# Main UI
show_main_interface()
show_sidebar(embedding_model, llm_model)

# File uploader
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type="pdf", 
    help=f"Upload a PDF document (max {MAX_FILE_SIZE_MB}MB) to analyze"
)

def process_document(file):
    if not embedding_model:
        st.error("Embedding model failed to load. Please refresh the page and try again.")
        return False
    
    try:
        with st.spinner("Processing document..."):
            timings = {}
            total_start = time.time()
            
            # Extract text
            extract_start = time.time()
            text, page_count = extract_text_from_pdf(file)
            extract_time = time.time() - extract_start
            timings['extraction'] = extract_time
            
            if not text:
                raise ValueError("Could not extract text from the PDF.")
            
            # Create chunks
            chunk_start = time.time()
            chunks = create_chunks(text, text_splitter)
            chunk_time = time.time() - chunk_start
            timings['chunking'] = chunk_time
            
            if not chunks:
                raise ValueError("Could not create text chunks from the document.")
            
            # Build index
            index_start = time.time()
            index, embeddings = build_search_index(chunks, embedding_model)
            index_time = time.time() - index_start
            timings['indexing'] = index_time
            
            if index is None:
                raise ValueError("Could not build search index.")
            
            total_time = time.time() - total_start
            timings['total'] = total_time
            
            # Update session state
            st.session_state.update({
                'chunks': chunks,
                'index': index,
                'embeddings': embeddings,
                'pdf_name': file.name,
                'pdf_text': text,
                'pdf_page_count': page_count,
                'document_processed': True,
                'processing_error': None,
                'processing_times': timings
            })
            
            logger.info(f"Document processed in {total_time:.2f} seconds")
            clear_gpu_memory()
            
        return True
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.session_state.processing_error = str(e)
        return False

# Document processing
if uploaded_file and not st.session_state.document_processed:
    try:
        if process_document(uploaded_file):
            st.success(f"Document '{uploaded_file.name}' processed successfully! {st.session_state.pdf_page_count} pages, {len(st.session_state.chunks)} chunks")
            times = st.session_state.processing_times
            st.caption(f"Processing time: {times['total']:.1f}s (Extract: {times['extraction']:.1f}s, "
                     f"Chunk: {times['chunking']:.1f}s, Index: {times['indexing']:.1f}s)")
        elif st.session_state.processing_error:
            st.error(f"Error processing document: {st.session_state.processing_error}")
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")

# Question answering
if st.session_state.document_processed:
    st.header(f"Ask questions about: {st.session_state.pdf_name}")
    show_document_stats()
    
    question = st.text_input("Enter your question about the document:",
                            help="Ask specific questions about the document content")
    
    num_results = 5
    search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if question and search_button:
        start_time = time.time()
        
        with st.spinner("Searching document..."):
            relevant_chunks = get_relevant_chunks(
                question, 
                embedding_model,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.rag_cache,
                top_k=num_results
            )
            
            if not relevant_chunks:
                st.warning("No relevant passages found for your question. Try rephrasing it or using different keywords.")
            else:
                if llm_model is not None:
                    answer = generate_response(
                        question, 
                        relevant_chunks,
                        llm_tokenizer,
                        llm_model,
                        st.session_state.rag_cache,
                        device
                    )
                    st.markdown("### Answer")
                    st.markdown(answer)
                else:
                    st.warning("Language model not available. Showing relevant passages only.")
                
                processing_time = time.time() - start_time
                st.caption(f"Query processed in {processing_time:.2f} seconds")

# Reset button
if st.session_state.document_processed:
    if st.button("Process a different document"):
        clear_gpu_memory()
        st.session_state.update({
            'document_processed': False,
            'chunks': [],
            'index': None,
            'embeddings': None,
            'pdf_name': "",
            'processing_error': None,
            'pdf_text': "",
            'pdf_page_count': 0,
            'rag_cache': {},
            'processing_times': {'extraction': 0, 'chunking': 0, 'indexing': 0, 'total': 0},
            'document_stats': {'doc_size': 0, 'chunk_size': 0, 'overlap': 0}
        })
        st.rerun()

if __name__ == "__main__":
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        st.sidebar.warning("⚠️ Running in CPU-only mode - performance will be limited")