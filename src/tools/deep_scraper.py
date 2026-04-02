""" Agentic On-Demand Deep Scraper Tool """

from langchain_core.tools import tool
from src.tools.scraper_tool import scrape_blog
from src.tools.vector_operations import vector_store
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@tool
async def fetch_url_context(url: str) -> str:
    """Fetch additional details from a specific Markdown URL mentioned in the text to answer the user's question. 
    Use this tool ONLY if you need to read the contents of a specific hyperlink to gather more information."""
    logger.info(f"Agent requested to fetch URL context via Tool: {url}")
    
    try:
        # We scrape the live Jina reader content to feed immediately back to the LLM
        content = await scrape_blog(url)
        
        if not content:
            return f"Error: Could not retrieve content from {url}. It might be invalid, protected, or not a webpage."
        
        # We store it in Qdrant for future users (Caching Win & Upgrade Problem solution)
        # But ONLY if it's not already in there.
        if not vector_store.check_blog_exists(url):
            logger.info(f"Adding newly fetched Deep Link {url} to Qdrant (depth=1).")
            # We don't have the parent's exact context here, so we map it as a generic Supplementary reference
            vector_store.add_blog_content(
                content=content,
                blog_url=url,
                blog_title=f"Supplementary Context: {url}",
                company_name="Extracted Reference",
                depth=1,
                parent_url=None
            )
        else:
            logger.info(f"Deep Link {url} already exists in Qdrant. Continuing to serve text to LLM.")

        # Return up to 30,000 characters to ensure we don't blow out the context window 
        # but still give the LLM plenty of deep reading material to answer the user.
        return f"Content of {url}:\n\n{content[:30000]}"
    except Exception as e:
        logger.error(f"Tool error fetching {url}: {e}")
        return f"Error: An exception occurred while fetching {url}: {e}"
