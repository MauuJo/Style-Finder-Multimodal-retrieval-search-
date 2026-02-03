import gradio as gr
import pandas as pd
import os
from tempfile import NamedTemporaryFile

# Import your modern 2026 modules
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
        # Save uploaded image to a temporary file for processing
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        image_path = temp_file.name
        image.save(image_path)
        
        try:
            # Step 1: Encode image for analysis
            user_encoding = self.image_processor.encode_image(image_path)
            
            # Step 2: Use Gemini to get a style description for vector search
            # We use the 'base64' string we created in the processor
            style_query = self.llm_service.model.invoke([
                "Describe this outfit in 5 search keywords (e.g. 'red floral dress').",
                user_encoding['image']
            ]).content

            # Step 3: Find closest match in Taylor Swift database
            # For simplicity in this lab, we use a search query or vector match
            closest_row, similarity_score = self.image_processor.find_closest_match(
                user_encoding['vector'], self.data
            )
            
            # Step 4: Retrieve all items in that outfit (Metadata)
            all_items = get_all_items_for_image(closest_row['Image URL'], self.data)
            
            # Step 5: Generate the Final Stylist Response
            final_output = self.llm_service.generate_fashion_response(
                user_image_base64=user_encoding['base64'],
                matched_row=closest_row,
                all_items=all_items,
                similarity_score=similarity_score
            )
            
            return process_response(final_output)

        finally:
            # Cleanup
            if os.path.exists(image_path):
                os.unlink(image_path)

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