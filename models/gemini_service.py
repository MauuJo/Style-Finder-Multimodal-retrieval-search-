import logging
import os
from dotenv import load_dotenv  # <--- Added to load environment variables
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables (gets GOOGLE_API_KEY from .env)
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiVisionService:
    """
    Interfaces with Gemini 1.5 Flash to provide professional fashion analysis.
    """
    
    def __init__(self, model_id="gemini-1.5-flash", temperature=0.2):
        """
        Initialize the Gemini service.
        """
        # Safety Check: Ensure the key is loaded
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please check your .env file.")

        self.model = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_output_tokens=2048,
            google_api_key=api_key  # <--- Explicitly passing the key
        )
        logger.info("Gemini Vision Service Initialized.")

    def generate_fashion_response(self, user_image_base64, matched_row, all_items, 
                                 similarity_score, threshold=0.8):
        """
        Generates a professional retail analysis using RAG (Retrieved data).
        """
        # 1. Prepare the 'Retrieved' context from our database
        items_list = []
        for _, row in all_items.iterrows():
            item_str = f"{row['Item Name']} (${row['Price']}): {row['Link']}"
            items_list.append(item_str)
        
        items_description = "\n".join([f"- {item}" for item in items_list])
        
        # 2. Determine if it's an 'Exact Match' or 'Similar Style'
        is_exact = similarity_score >= threshold
        section_header = "ITEM DETAILS:" if is_exact else "SIMILAR ITEMS:"
        
        # 3. Construct the Multimodal Prompt
        system_context = (
            "You are a professional retail catalog analyst. Analyze the outfit in the image "
            "using formal, clinical language. Use the provided metadata to identify specific items."
        )
        
        prompt = f"""
        {system_context}
        
        Instructions:
        1. Identify the garments (colors, patterns, materials).
        2. Categorize the overall style aesthetic.
        3. ALWAYS include the {section_header} section at the end using the metadata below.
        
        METADATA FROM CATALOG:
        {items_description}
        """

        # 4. Create the multimodal message (Image + Text)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{user_image_base64}"}
                },
            ]
        )

        try:
            logger.info("Sending request to Gemini...")
            response = self.model.invoke([message])
            content = response.content
            
            # 5. Safety Check: Ensure the links are actually there
            if section_header not in content:
                logger.warning("Links section missing, appending manually.")
                content += f"\n\n## {section_header}\n{items_description}"
                
            return content
            
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            return f"Error: {e}. Fallback Info:\n{items_description}"