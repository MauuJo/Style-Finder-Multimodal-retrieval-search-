import logging
import re
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_all_items_for_image(image_url: str, dataset: pd.DataFrame):
    """
    Finds every fashion item (shoes, dress, jewelry) associated with the 
    matched image in our catalog.
    """
    # Filter the DataFrame for the specific outfit URL
    related_items = dataset[dataset['Image URL'] == image_url]
    logger.info(f"Retrieved {len(related_items)} distinct items for outfit: {image_url}")
    return related_items

def format_alternatives_response(user_response, alternatives, similarity_score, threshold=0.8):
    """
    Adds a 'Similar Items' section with clickable links to the AI's analysis.
    """
    # Safety Check: If Gemini gave a generic refusal, we provide a clean header
    refusal_phrases = ["I cannot", "I am unable", "I'm not able", "policy"]
    if not user_response or any(p in user_response.lower() for p in refusal_phrases):
        user_response = "## Fashion Analysis Results\n\nI've analyzed your image and found matching styles in our catalog:"

    # Determine Header based on match quality
    header = "## Exact Matches Found" if similarity_score >= threshold else "## Recommended Similar Styles"
    enhanced_response = f"{user_response}\n\n{header}\n"

    # Limit the number of alternative items to keep the UI clean
    items_added = 0
    max_items = 6

    for item, alts in alternatives.items():
        enhanced_response += f"\n### {item}\n"
        if alts:
            for alt in alts[:2]: # Show up to 2 variations per item
                if items_added < max_items:
                    # Format as a clean Markdown list with bolded prices
                    enhanced_response += f"- **{alt['title']}** — **{alt['price']}** at {alt['source']} ([View Item]({alt['link']}))\n"
                    items_added += 1
        else:
            enhanced_response += "- *Currently no direct matches in the local catalog.*\n"

    return enhanced_response

def process_response(response: str) -> str:
    """
    Cleans up the LLM output (escaping symbols, fixing headers) for the Gradio UI.
    """
    if not response:
        return "# Style Analysis\n\nNo visual data could be parsed. Please try a clearer image."

    # Fix character issues (like $ signs breaking Markdown)
    processed = response.replace("$", "\\$")

    # Clean up Section Headers if the AI used "ITEM DETAILS:" instead of MarkDown
    processed = processed.replace("ITEM DETAILS:", "## Item Details")
    processed = processed.replace("SIMILAR ITEMS:", "## Similar Items")

    # Ensure consistent bullet points
    processed = re.sub(r'^\* ', '- ', processed, flags=re.MULTILINE)

    return processed