import torch
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import requests
from sklearn.metrics.pairwise import cosine_similarity
# We use the latest LangChain-Google integration
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv() 

class ImageProcessor:
    def __init__(self):
        # Ensure the API key is set in the environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")
            
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key # Explicitly pass the key
        )
        print("Multimodal Image Processor Initialized.")

    def encode_image(self, image_input, is_url=False):
        """
        Converts image to Base64 for Gemini Vision and extracts 
        visual features for the Vector Database.
        """
        try:
            if is_url:
                response = requests.get(image_input)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(image_input).convert("RGB")

            # 1. Create Base64 for the LLM 'Style Analysis'
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 2. In a real generalized system, we'd use a vision encoder here.
            # For this lab, we'll return the base64 and use it to get 
            # textual descriptions for our vector search.
            return {"base64": base64_string, "image": img}
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def find_closest_match(self, query_vector, dataset):
        """
        Finds the closest fashion item in our .pkl database.
        """
        try:
            # We stack the pre-computed embeddings from the Taylor Swift dataset
            dataset_vectors = np.vstack(dataset['Embedding'].values)
            
            # Use Cosine Similarity to find the best match
            similarities = cosine_similarity(query_vector.reshape(1, -1), dataset_vectors)
            
            index = np.argmax(similarities)
            score = similarities[0][index]
            
            return dataset.iloc[index], score
        except Exception as e:
            print(f"Search Error: {e}")
            return None, 0