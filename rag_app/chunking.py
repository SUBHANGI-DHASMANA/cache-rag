# Text splitting and chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
from .config import logger, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, get_optimal_chunk_params

@st.cache_resource
def load_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP
    )

def create_chunks(text: str, text_splitter: RecursiveCharacterTextSplitter) -> List[str]:
    try:
        chunk_size, chunk_overlap = get_optimal_chunk_params(len(text))
        text_splitter.chunk_size = chunk_size
        text_splitter.chunk_overlap = chunk_overlap
        
        status_text = st.empty()
        status_text.text("Splitting document into chunks...")
        
        if len(text) > 200000:
            sections = [text[i:i+100000] for i in range(0, len(text), 100000)]
            all_chunks = []
            
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(
                    lambda x: text_splitter.split_documents([Document(page_content=x)]), 
                    section
                ) for section in sections]
                
                for i, future in enumerate(as_completed(futures)):
                    section_chunks = future.result()
                    all_chunks.extend([
                        doc.page_content for doc in section_chunks 
                        if len(doc.page_content.strip()) > 20
                    ])
                    status_text.text(f"Processing section {i+1}/{len(sections)}...")
                    
            chunks = all_chunks
        else:
            documents = [Document(page_content=text)]
            docs = text_splitter.split_documents(documents)
            chunks = [doc.page_content for doc in docs if len(doc.page_content.strip()) > 20]
            
        status_text.empty()
        logger.info(f"Created {len(chunks)} chunks with size {chunk_size} and overlap {chunk_overlap}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        if 'status_text' in locals():
            status_text.empty()
        raise RuntimeError(f"Failed to create document chunks: {str(e)}")