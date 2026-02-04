import gradio as gr
import pandas as pd
import os
import json  # <--- NEW IMPORT for parsing Gemini's output
from tempfile import NamedTemporaryFile
from langchain_core.messages import HumanMessage

# Import local modules
from models.image_processor import ImageProcessor
from models.gemini_service import GeminiVisionService
from utils.helpers import get_all_items_for_image, process_response
from utils.web_search import search_live_products
import config

class StyleFinderApp:
    def __init__(self, dataset_path):
        """
        Initializes the Style Finder with the vector database.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset missing at {dataset_path}")
            
        self.data = pd.read_pickle(dataset_path)
        
        # Initialize Gemini and Image Processing components
        self.image_processor = ImageProcessor()
        self.llm_service = GeminiVisionService(
            model_id=config.MODEL_ID,
            temperature=config.TEMPERATURE
        )

    def process_image(self, image):
        """
        Dynamic Hybrid RAG:
        1. Encodes image.
        2. Asks Gemini to detect ANY fashion categories (Top, Shoes, Bag, etc.).
        3. Loops through every detected category and searches the live web for it.
        4. Merges everything with the Taylor Swift database results.
        """
        # Save uploaded image to a temporary file
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close()
        image_path = temp_file.name
        image.save(image_path)
        
        try:
            # Step 1: Encode Image (Get Base64 + Vector)
            user_encoding = self.image_processor.encode_image(image_path)
            
            # Step 2: Ask Gemini to Detect Categories (The "Agent" Step)
            keyword_prompt = HumanMessage(content=[
                {"type": "text", "text": """
                Analyze this outfit and identify ALL distinct fashion items visible (e.g., top, bottom, footwear, bag, jewelry, outerwear, eyewear).
                
                Return ONLY a raw JSON object where:
                - Keys are the category name (e.g., "footwear", "handbag", "earrings").
                - Values are the specific shopping search query for India (e.g., "silver rhinestone ankle boots buy online").
                
                Example:
                {
                    "top": "sequin bodysuit silver long sleeve",
                    "bottom": "burgundy high waist shorts",
                    "footwear": "black ankle boots block heel",
                    "jewelry": "diamond drop earrings"
                }
                Do not use markdown formatting.
                """},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_encoding['base64']}"}},
            ])
            
            # Parse Gemini's Response
            raw_response = self.llm_service.model.invoke([keyword_prompt]).content
            cleaned_json = raw_response.replace("```json", "").replace("```", "").strip()
            
            search_queries = {}
            try:
                search_queries = json.loads(cleaned_json)
                print(f"🤖 Detected Categories: {list(search_queries.keys())}")
            except Exception as e:
                print(f"❌ JSON Parse Error: {e}")
                # Fallback search if JSON parsing fails
                search_queries = {"outfit": "trending fashion style match"}

            # Step 3: Find closest match in Taylor Swift DB (Static RAG)
            closest_row, similarity_score = self.image_processor.find_closest_match(
                user_encoding['vector'], self.data
            )
            all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
            
            # Step 4: Generate Base Analysis from Gemini
            final_output = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding['base64'],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score
            )
            
            # Step 5: Dynamic Loop for Live Search Results
            final_output += "\n\n# 🛍️ Get The Look (Live India Search)\n"
            
            # Loop through detected categories (Limit to 5 to avoid timeouts)
            for category, query in list(search_queries.items())[:5]:
                print(f"🔎 Processing Category: {category}...")
                results = search_live_products(query)
                
                if results:
                    # Capitalize category (e.g., "footwear" -> "Footwear")
                    final_output += f"\n### 🔍 {category.title()}\n"
                    for item in results[:2]: # Show top 2 results per category
                        final_output += f"- [{item['title']}]({item['link']})\n"
            
            return process_response(final_output)

        finally:
            if os.path.exists(image_path):
                try: os.unlink(image_path)
                except: pass

def create_gradio_interface(app):
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Fashion Stylist") as demo:
        gr.Markdown("# 👗 Hybrid Style Finder\nUpload an outfit to get a professional analysis + Live Shopping Links from India.")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Outfit")
                submit_btn = gr.Button("Analyze Style", variant="primary")
            with gr.Column():
                output = gr.Markdown(label="Stylist Recommendations")
        
        submit_btn.click(fn=app.process_image, inputs=image_input, outputs=output)
        
    return demo

if __name__ == "__main__":
    app = StyleFinderApp("swift-style-embeddings.pkl")
    demo = create_gradio_interface(app)
    demo.launch()