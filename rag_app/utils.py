# Utility functions
import torch
import gc
import logging
from typing import Optional
from .config import logger

def clear_gpu_memory():
    try:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU memory cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")

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