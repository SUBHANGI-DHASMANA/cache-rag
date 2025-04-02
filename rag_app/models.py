# Model loading and management
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Tuple, Optional, Any
from .config import logger, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME
from .utils import clear_gpu_memory


@st.cache_resource
def load_embedding_model(device: torch.device) -> Optional[SentenceTransformer]:
    clear_gpu_memory()
    try:
        logger.info("Loading embedding model...")
        if str(device) == "mps":
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            model.max_seq_length = 384
        else:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=str(device))
            model.max_seq_length = 512
        logger.info(f"Embedding model loaded on {model.device}")
        return model
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None

@st.cache_resource
def load_llm_model(device: torch.device) -> Tuple[Optional[Any], Optional[Any]]:
    clear_gpu_memory()
    try:
        logger.info("Loading LLM model...")
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        
        dtype = torch.float16 if str(device) in ["mps", "cuda"] else torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={str(device): "4GiB", "cpu": "8GiB"}
        )
        logger.info("LLM model loaded successfully")
        return tokenizer, model
    except Exception as e:
        logger.error(f"Error loading language model: {e}")
        return None, None