# Configuration and constants
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_FILE_SIZE_MB = 100
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

def get_optimal_chunk_params(doc_length: int) -> Tuple[int, int]:
    if doc_length < 10000:
        return 500, 100
    elif doc_length < 50000:
        return 750, 150
    elif doc_length < 200000:
        return 1000, 200
    else:
        return 1500, 300