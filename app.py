import gradio as gr
import pandas as pd
import os
from tempfile import NamedTemporaryFile
from langchain_core.messages import HumanMessage  # <--- Added Import

# Import local modules
from models.image_processor import ImageProcessor
from models.gemini_service import GeminiVisionService
from utils.helpers import get_all_items_for_image, process_response
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
        The core RAG pipeline:
        1. Encode -> 2. Retrieve -> 3. Augment -> 4. Generate
        """
        # --- FIX FOR WINDOWS PERMISSION ERROR ---
        # We must close the file handle immediately so it doesn't get locked
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.close() 
        
        image_path = temp_file.name
        image.save(image_path)
        
        try:
            # Step 1: Encode image for analysis
            user_encoding = self.image_processor.encode_image(image_path)
            
            # --- FIX FOR LANGCHAIN ERROR ---
            # We must use HumanMessage with the Base64 string, not the PIL Image object
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe this outfit in 5 search keywords (e.g. 'red floral dress')."},
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{user_encoding['base64']}"}
                    },
                ]
            )
            
            style_query = self.llm_service.model.invoke([message]).content

            # Step 2: Find closest match in Taylor Swift database
            closest_row, similarity_score = self.image_processor.find_closest_match(
                user_encoding['vector'], self.data
            )
            
            # Step 3: Retrieve all items in that outfit (Metadata)
            all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
            
            # Step 4: Generate the Final Stylist Response
            final_output = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding['base64'],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score
            )
            
            return process_response(final_output)

        finally:
            # Cleanup: Now this should work because we closed the file handle earlier
            if os.path.exists(image_path):
                try:
                    os.unlink(image_path)
                except Exception as e:
                    print(f"Warning: Could not delete temp file: {e}")

def create_gradio_interface(app):
    with gr.Blocks(theme=gr.themes.Soft(), title="AI Fashion Stylist") as demo:
        gr.Markdown("# 👗 Generalized Style Finder\nUpload any outfit to get a professional analysis and see similar items from our catalog.")
        
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