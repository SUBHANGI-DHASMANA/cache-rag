import streamlit as st
import time
import os
import numpy as np
import PyPDF2
import logging
import gc
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from usearch.index import Index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions for memory management
def clear_gpu_memory():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")

def get_gpu_stats() -> Dict[str, float]:
    stats = {'allocated': 0, 'total': 0, 'utilization': 0}
    
    try:
        if torch.backends.mps.is_available():
            stats['allocated'] = torch.mps.current_allocated_memory() / (1024 * 1024)
            stats['total'] = torch.mps.driver_allocated_memory() / (1024 * 1024)
            stats['utilization'] = (stats['allocated'] / stats['total']) * 100 if stats['total'] > 0 else 0
    except Exception as e:
        logger.warning(f"Could not get GPU stats: {e}")
    
    return stats

def setup_device() -> torch.device:
    try:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Metal) device for Apple Silicon")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device
    except Exception as e:
        logger.warning(f"Error setting up device, falling back to CPU: {e}")
        return torch.device("cpu")

device = setup_device()

@st.cache_resource
def load_embedding_model() -> Optional[SentenceTransformer]:
    clear_gpu_memory()
    
    try:
        logger.info("Loading embedding model...")
        if str(device) == "mps":
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded on CPU (MPS compatibility mode)")
        else:
            model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))
            logger.info(f"Embedding model loaded on {model.device}")
        if str(device) == "mps":
            model.max_seq_length = 384 
        else:
            model.max_seq_length = 512 
            
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        st.error(f"Failed to load embedding model: {str(e)}. Please refresh and try again.")
        return None

@st.cache_resource
def load_llm_model() -> Tuple[Optional[Any], Optional[Any]]:
    clear_gpu_memory() 
    try:
        logger.info("Loading LLM model...")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        if str(device) == "mps":
            dtype = torch.float16
        elif str(device) == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={str(device): "4GiB", "cpu": "8GiB"}
        )
        logger.info(f"LLM model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading language model: {e}")
        st.error(f"Failed to load language model: {str(e)}. Functionality will be limited.")
        return None, None

def get_optimal_chunk_params(doc_length: int) -> Tuple[int, int]:
    if doc_length < 10000:
        return 500, 100
    elif doc_length < 50000:
        return 750, 150
    elif doc_length < 200000:
        return 1000, 200
    else:
        return 1500, 300

@st.cache_resource
def load_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

try:
    embedding_model = load_embedding_model()
    llm_tokenizer, llm_model = load_llm_model() if embedding_model is not None else (None, None)
    text_splitter = load_text_splitter()
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    embedding_model = None
    llm_tokenizer, llm_model = None, None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

st.title("📄 Cache RAG PDF Q&A")
st.markdown("""
Upload a PDF document and ask questions about its content.
""")

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
        'gpu_stats': {'allocated': 0, 'total': 0, 'utilization': 0},
        'processing_times': {'extraction': 0, 'chunking': 0, 'indexing': 0, 'total': 0},
        'document_stats': {'doc_size': 0, 'chunk_size': 0, 'overlap': 0}
    })

max_size_mb = 100
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type="pdf", 
    help=f"Upload a PDF document (max {max_size_mb}MB) to analyze"
)

def extract_text_from_pdf(file) -> Tuple[str, int]:
    text = ""
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > max_size_mb * 1024 * 1024:
            raise ValueError(f"File size exceeds {max_size_mb}MB limit")
        
        pdf_reader = PyPDF2.PdfReader(file)
        page_count = len(pdf_reader.pages)
        
        if page_count == 0:
            raise ValueError("The PDF has no pages")
        progress_bar = st.progress(0)
        status_text = st.empty()
        def extract_page_text(page_num):
            try:
                page = pdf_reader.pages[page_num]
                return page.extract_text() or ""
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num}: {e}")
                return ""
        with ThreadPoolExecutor(max_workers=min(10, page_count)) as executor:
            futures = {executor.submit(extract_page_text, i): i for i in range(page_count)}
            pages = [""] * page_count
            
            for i, future in enumerate(as_completed(futures)):
                page_num = futures[future]
                pages[page_num] = future.result()
                progress = (i + 1) / page_count
                progress_bar.progress(progress)
                status_text.text(f"Extracting text: {i+1}/{page_count} pages")
        
        text = "\n\n".join(pages)
        progress_bar.empty()
        status_text.empty()
        return text, page_count
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.empty()
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

def create_chunks(text: str) -> List[str]:
    try:
        chunk_size, chunk_overlap = get_optimal_chunk_params(len(text))
        st.session_state.document_stats.update({
            'doc_size': len(text),
            'chunk_size': chunk_size,
            'overlap': chunk_overlap
        })
        text_splitter.chunk_size = chunk_size
        text_splitter.chunk_overlap = chunk_overlap
        status_text = st.empty()
        status_text.text("Splitting document into chunks...")
        if len(text) > 200000:
            sections = [text[i:i+100000] for i in range(0, len(text), 100000)]
            all_chunks = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(lambda x: text_splitter.split_documents(
                    [Document(page_content=x)]), section) for section in sections]
                for i, future in enumerate(as_completed(futures)):
                    section_chunks = future.result()
                    all_chunks.extend([doc.page_content for doc in section_chunks 
                                      if len(doc.page_content.strip()) > 20])
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

def build_search_index(chunks: List[str]) -> Tuple[Optional[Index], Optional[np.ndarray]]:
    if not chunks:
        raise ValueError("No chunks provided to build index")
    
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Generating embeddings...")
        dim = embedding_model.get_sentence_embedding_dimension()
        index = Index(ndim=dim, metric='cos')
        gpu_stats = get_gpu_stats()
        batch_size = 32 if gpu_stats['utilization'] > 50 else 64
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        all_embeddings = []
        
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            progress = (i + 1) / num_batches
            progress_bar.progress(progress)
            status_text.text(f"Generating embeddings (batch {i+1}/{num_batches})...")
            if torch.backends.mps.is_available() and get_gpu_stats()['utilization'] > 80:
                clear_gpu_memory()
            batch_embeddings = embedding_model.encode(
                batch_chunks,
                show_progress_bar=False,
                device=str(device),
                convert_to_numpy=True
            )
            
            all_embeddings.append(batch_embeddings)
            if i % 5 == 0 and torch.backends.mps.is_available():
                clear_gpu_memory()
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

def process_document(file):
    if not embedding_model:
        st.error("Embedding model failed to load. Please refresh the page and try again.")
        return False
    
    try:
        with st.spinner("Processing document..."):
            timings = {}
            total_start = time.time()
            extract_start = time.time()
            text, page_count = extract_text_from_pdf(file)
            extract_time = time.time() - extract_start
            timings['extraction'] = extract_time
            
            if not text:
                raise ValueError("Could not extract text from the PDF.")
            chunk_start = time.time()
            chunks = create_chunks(text)
            chunk_time = time.time() - chunk_start
            timings['chunking'] = chunk_time
            
            if not chunks:
                raise ValueError("Could not create text chunks from the document.")
            index_start = time.time()
            index, embeddings = build_search_index(chunks)
            index_time = time.time() - index_start
            timings['indexing'] = index_time
            
            if index is None:
                raise ValueError("Could not build search index.")
            total_time = time.time() - total_start
            timings['total'] = total_time
            
            st.session_state.update({
                'chunks': chunks,
                'index': index,
                'embeddings': embeddings,
                'pdf_name': file.name,
                'pdf_text': text,
                'pdf_page_count': page_count,
                'document_processed': True,
                'processing_error': None,
                'rag_cache': {},
                'processing_times': timings
            })
            logger.info(f"Document processed in {total_time:.2f} seconds")
            logger.info(f"Performance breakdown: Extract={extract_time:.2f}s, "
                       f"Chunk={chunk_time:.2f}s, Index={index_time:.2f}s")
            clear_gpu_memory()
            
        return True
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        st.session_state.processing_error = str(e)
        return False

def get_relevant_chunks(query: str, top_k: int = 5) -> List[str]:
    if not query.strip():
        return []
    
    try:
        cache_key = f"query_result_{hash(query)}_{top_k}"
        if cache_key in st.session_state.rag_cache:
            return st.session_state.rag_cache[cache_key]
        query_embedding = embedding_model.encode(
            query,
            device=str(device),
            convert_to_numpy=True
        )
        try:
            matches = st.session_state.index.search(np.array(query_embedding), top_k)
            result_chunks = [st.session_state.chunks[i] for i in matches.keys 
                           if i < len(st.session_state.chunks)]
        except Exception as search_error:
            logger.error(f"Error in vector search: {search_error}")
            result_chunks = []
            query_terms = set(query.lower().split())
            chunk_scores = []
            
            for i, chunk in enumerate(st.session_state.chunks):
                chunk_lower = chunk.lower()
                score = sum(1 for term in query_terms if term in chunk_lower)
                chunk_scores.append((score, i))
            chunk_scores.sort(reverse=True)
            result_chunks = [st.session_state.chunks[i] for _, i in chunk_scores[:top_k]]
        st.session_state.rag_cache[cache_key] = result_chunks
        return result_chunks
    except Exception as e:
        logger.error(f"Error finding relevant chunks: {e}")
        return []

def generate_response(query: str, context: List[str]) -> str:
    if not query.strip() or not context:
        return "I need both a question and context to provide an answer."
    
    if llm_model is None or llm_tokenizer is None:
        return "The language model is not available. I can show you relevant passages but cannot generate a comprehensive answer."
    
    try:
        cache_key = f"response_{hash(query)}_{hash(tuple([c[:100] for c in context]))}"
        if cache_key in st.session_state.rag_cache:
            return st.session_state.rag_cache[cache_key]
        formatted_context = ""
        for i, chunk in enumerate(context):
            chunk_preview = chunk[:min(len(chunk), 1500)]  
            formatted_context += f"\nDocument excerpt {i+1}:\n{chunk_preview.strip()}\n"
        prompt = f"""<s>[INST] You are an expert document analysis AI that answers questions based on provided document excerpts.
        
        Relevant document excerpts:
        {formatted_context}
        
        Question: {query}
        
        Instructions:
        1. Answer ONLY based on the provided document.
        2. Be concise but thorough, focusing on the most relevant information.
        3. If the answer isn't in the excerpts, explain that you don't have enough information.
        4. Do not make up information or draw from external knowledge.
        5. Include specific details from the document where relevant.
        [/INST]"""
        inputs = llm_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        is_complex = any(word in query.lower() for word in 
                         ['explain', 'analyze', 'compare', 'contrast', 'why', 'how does'])
        
        max_tokens = 350 if is_complex else 256
        temperature = 0.7 if is_complex else 0.5
        
        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=llm_tokenizer.eos_token_id,
                num_return_sequences=1,
                early_stopping=True
            )
        full_response = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        final_response = full_response.split("[/INST]")[-1].strip()
        if final_response.lower().startswith("answer:"):
            final_response = final_response[7:].strip()
        st.session_state.rag_cache[cache_key] = final_response
        if torch.backends.mps.is_available():
            clear_gpu_memory()
        return final_response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "I encountered an error while generating a response. This might be due to memory constraints. You can still view the relevant passages below."

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
        logger.error(f"Unexpected error in document processing: {e}")

if st.session_state.document_processed:
    st.header(f"Ask questions about: {st.session_state.pdf_name}")
    
    with st.expander("Document Stats", expanded=False):
        st.markdown(f"**Pages:** {st.session_state.pdf_page_count}")
        st.markdown(f"**Text chunks:** {len(st.session_state.chunks)}")
        doc_stats = st.session_state.document_stats
        st.markdown(f"**Document size:** {doc_stats['doc_size']:,} characters")
        st.markdown(f"**Chunk size:** {doc_stats['chunk_size']} characters with {doc_stats['overlap']} overlap")
        if len(st.session_state.pdf_text) > 0:
            st.markdown("**Document preview:**")
            st.text(st.session_state.pdf_text[:500] + "..." if len(st.session_state.pdf_text) > 500 else st.session_state.pdf_text)
    question = st.text_input("Enter your question about the document:",
                            help="Ask specific questions about the document content")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        num_results = st.slider("Passages to retrieve", 1, 10, 5,
                               help="More passages may provide better context but take longer to process")
    with col2:
        show_passages = st.checkbox("Show relevant passages", value=True,
                                   help="Display the document passages used to answer your question")
    with col3:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    if question and search_button:
        start_time = time.time()
        
        with st.spinner("Searching document..."):
            gpu_before = get_gpu_stats()
            relevant_chunks = get_relevant_chunks(question, top_k=num_results)
            
            if not relevant_chunks:
                st.warning("No relevant passages found for your question. Try rephrasing it or using different keywords.")
            else:
                if llm_model is not None:
                    answer = generate_response(question, relevant_chunks)
                    st.markdown("### Answer")
                    st.markdown(answer)
                else:
                    st.warning("Language model not available. Showing relevant passages only.")
                if show_passages:
                    with st.expander("Relevant Passages", expanded=True):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"**Passage {i+1}**")
                            display_chunk = chunk
                            for term in question.lower().split():
                                if len(term) > 3:  
                                    display_chunk = display_chunk.replace(term, f"**{term}**")
                            st.markdown(display_chunk[:800] + ("..." if len(chunk) > 800 else ""))
                gpu_after = get_gpu_stats()
                processing_time = time.time() - start_time
                st.caption(f"Query processed in {processing_time:.2f} seconds | "
                          f"GPU Memory: {gpu_after['allocated']:.1f}/{gpu_after['total']:.1f} MB")
                
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

with st.sidebar:
    st.markdown("## System Monitor")
    if torch.backends.mps.is_available():
        gpu_stats = get_gpu_stats()
        gpu_col1, gpu_col2 = st.columns(2)
        with gpu_col1:
            st.metric("GPU Memory Used", 
                    f"{gpu_stats['allocated']:.1f} MB",
                    help="Currently allocated GPU memory")
        with gpu_col2:
            st.metric("Memory Available", 
                    f"{gpu_stats['total']:.1f} MB",
                    help="Total available GPU memory")
        st.progress(min(gpu_stats['allocated'] / gpu_stats['total'], 1.0), 
                   text=f"GPU Usage: {min(gpu_stats['allocated']/gpu_stats['total'],1.0)*100:.0f}%")
        if st.button("Clear GPU Memory", help="Free up GPU memory if performance decreases"):
            clear_gpu_memory()
            new_stats = get_gpu_stats()
            st.success(f"GPU memory cleared! ({new_stats['allocated']:.1f} MB now in use)")
    else:
        st.warning("Running on CPU - GPU acceleration not available")

    st.markdown("## Model Information")
    if embedding_model:
        st.markdown(f"**Embedding:** all-MiniLM-L6-v2 (dim={embedding_model.get_sentence_embedding_dimension()})")
    if llm_model:
        st.markdown(f"**LLM:** DeepSeek-R1-Distill-Qwen-1.5B")
    st.markdown(f"**Device:** {device}")
    
    st.markdown("## Cache Management")
    if 'rag_cache' in st.session_state:
        cache_entries = len(st.session_state.rag_cache)
        st.markdown(f"**Cached responses:** {cache_entries}")
        if st.button("Clear Response Cache"):
            st.session_state.rag_cache = {}
            st.success("Response cache cleared")
    
    st.markdown("---")
    if st.button("Reset Entire App", type="secondary"):
        st.cache_resource.clear()
        st.session_state.clear()
        clear_gpu_memory()
        st.rerun()

if __name__ == "__main__":
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        st.sidebar.warning("⚠️ Running in CPU-only mode - performance will be limited")
    if torch.backends.mps.is_available():
        current_mem = torch.mps.current_allocated_memory() / (1024 * 1024)
        st.sidebar.caption(f"Current memory usage: {current_mem:.1f} MB")