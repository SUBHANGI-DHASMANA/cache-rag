# UI components
import streamlit as st
from typing import Dict, Any

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.update({
            'initialized': True,
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

def show_sidebar(embedding_model, llm_model):
    with st.sidebar:
        st.markdown("## Model Information")
        if embedding_model:
            st.markdown(f"**Embedding:** all-MiniLM-L6-v2 (dim={embedding_model.get_sentence_embedding_dimension()})")
        if llm_model:
            st.markdown(f"**LLM:** DeepSeek-R1-Distill-Qwen-1.5B")
        
        st.markdown("## Cache Management")
        if 'rag_cache' in st.session_state:
            cache_entries = len(st.session_state.rag_cache)
            st.markdown(f"**Cached responses:** {cache_entries}")
            if st.button("Clear Response Cache"):
                st.session_state.rag_cache = {}
                st.success("Response cache cleared")

def show_document_stats():
    with st.expander("Document Stats", expanded=False):
        st.markdown(f"**Pages:** {st.session_state.pdf_page_count}")
        st.markdown(f"**Text chunks:** {len(st.session_state.chunks)}")
        doc_stats = st.session_state.document_stats
        st.markdown(f"**Document size:** {doc_stats['doc_size']:,} characters")
        st.markdown(f"**Chunk size:** {doc_stats['chunk_size']} characters with {doc_stats['overlap']} overlap")

def show_main_interface():
    st.title("📄 Cache RAG PDF Q&A")
    st.markdown("Upload a PDF document and ask questions about its content.")