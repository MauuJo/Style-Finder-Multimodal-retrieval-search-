import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import base64
from io import BytesIO
import requests
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

class ImageProcessor:
    def __init__(self):
        """
        Hybrid Processor:
        - Uses FULL ResNet50 (1000 dims) to match the Taylor Swift dataset.
        - Prepares Base64 for Gemini Vision analysis.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Pre-trained ResNet50
        weights = models.ResNet50_Weights.DEFAULT
        self.model = models.resnet50(weights=weights)
        
        # --- FIX: USE FULL MODEL TO MATCH DATASET (1000 DIMS) ---
        # We do NOT remove the last layer anymore. 
        # The dataset expects the 1000-dimensional output.
        self.model = self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet normalization
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("Image Processor (1000-dim ResNet + Gemini) Initialized.")

    def encode_image(self, image_path, is_url=False):
        """
        Processes image to return BOTH a Vector (for search) and Base64 (for AI).
        """
        try:
            # Load Image
            if is_url:
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img = Image.open(image_path).convert("RGB")

            # 1. Prepare for Gemini (Base64)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 2. Prepare for Vector DB (ResNet50 Vector)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # This will now produce a [1, 1000] vector
                vector = self.model(img_tensor).cpu().numpy().flatten()

            return {
                "base64": base64_string, 
                "vector": vector,
                "image": img
            }
            
        except Exception as e:
            print(f"Encoding Error: {e}")
            return {"base64": None, "vector": None}

    def find_closest_match(self, query_vector, dataset):
        """
        Finds the closest image in the dataset using Cosine Similarity.
        """
        try:
            # Stack all dataset vectors into a big matrix
            dataset_vectors = np.vstack(dataset['Embedding'].values)
            
            # Compare user's vector vs dataset
            # Both should now be (1, 1000) and (N, 1000)
            similarities = cosine_similarity(query_vector.reshape(1, -1), dataset_vectors)
            
            # Get best match
            index = np.argmax(similarities)
            score = similarities[0][index]
            
            return dataset.iloc[index], score
        except Exception as e:
            print(f"Search Error: {e}")
            return None, 0