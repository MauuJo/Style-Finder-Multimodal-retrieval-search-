"""
Configuration settings for the Generalized Style Finder application (2026).
"""
import os
from dotenv import load_dotenv

# Load environment variables (like GOOGLE_API_KEY)
load_dotenv()

# --- Gemini Model Configuration ---
# We use Gemini 1.5 Flash: It is multimodal, extremely fast, and 
# perfect for analyzing fashion images.
MODEL_ID = "gemini-2.5-flash"
TEMPERATURE = 0.2  # Low temperature for factual, consistent style analysis

# --- Image Processing Settings ---
# These are standard values used by most vision models (ViT/CLIP)
IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# --- RAG & Search Settings ---
# Threshold for finding a 'close match' in our vector database
SIMILARITY_THRESHOLD = 0.8
# Number of similar items to pull from the Taylor Swift catalog
DEFAULT_ALTERNATIVES_COUNT = 3

# --- Generalized System Prompt ---
# This ensures Gemini behaves like a stylist, not just a labeler.
SYSTEM_PROMPT = """
You are a professional fashion stylist and visual analyst. 
When shown an image, your task is to:
1. Identify the core items (e.g., 'silk midi dress', 'leather ankle boots').
2. Identify the aesthetic 'vibe' (e.g., 'quiet luxury', 'streetwear', 'cottagecore').
3. Extract specific search attributes (color, pattern, material, neckline).

Your analysis will be used to search a catalog for similar items. 
Be precise and descriptive.
"""