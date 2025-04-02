# PDF processing and text extraction
import os
import PyPDF2
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from .config import logger, MAX_FILE_SIZE_MB

def extract_text_from_pdf(file) -> Tuple[str, int]:
    text = ""
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB limit")
        
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