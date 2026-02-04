from duckduckgo_search import DDGS

def search_live_products(query_text, num_results=3):
    """
    Searches for a specific fashion item in India using DuckDuckGo.
    
    Args:
        query_text (str): The specific item to search (e.g., "silver sequin top")
        num_results (int): Number of links to return per item.
    """
    # Safety check for empty queries
    if not query_text or len(query_text) < 3:
        return []

    # Ensure we target Indian shopping sites if not already present
    if "india" not in query_text.lower():
        query_text += " india buy online"

    print(f"🔎 Searching DDG for: '{query_text}'")
    
    results = []
    try:
        # region="in-en" targets India
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query_text, 
                region="in-en", 
                max_results=num_results
            )
            
            for res in search_results:
                results.append({
                    "title": res['title'],
                    "link": res['href']
                })
        return results

    except Exception as e:
        print(f"❌ Web Search Error for '{query_text}': {e}")
        return []