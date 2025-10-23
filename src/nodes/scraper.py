""" Scraper Node - Extract blog content """

from langchain_core.messages import AIMessage
from src.state import AgentState
from src.tools.scraper_tool import scrape_blog
from src.tools.vector_operations import vector_store
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def scraper_node(state: AgentState) -> AgentState:
    """
    Scrape blog content and store in vector database
    """

    logger.info("=== SCRAPER NODE ===")
    
    blog_url = state.get("selected_blog_url")
    blog_title = state.get('selected_blog_title')
    company_name = state.get("company_name")
    if not blog_url:    
        logger.info("no blog url found")
    reason = state.get("scraper_reason", "Requested by agent")

    logger.info(f"Scraping blog: {blog_url}")
    logger.info(f"Reason: {reason}")

    new_state = state.copy()

    # Always scrape the content to ensure it's fresh
    content = await scrape_blog(blog_url)

    if not content:
        logger.error(f"Failed to scrape content from {blog_url}")
        new_state.update({
            "messages": state['messages'] + [AIMessage(content="I had trouble accessing that blog. Could you try another one?")],
            "should_scrape": False,
            "step_count": state['step_count'] + 1
        })
        return new_state

    logger.info(f"Storing content in vector database: {len(content)} characters")
    point_ids = vector_store.add_blog_content(
        content=content,
        blog_url=blog_url,
        blog_title=blog_title,
        company_name=company_name
    )

    logger.info(f"Successfully stored {len(point_ids)} chunks in vector store")
    new_state.update({
        "vector_store_ids": state.get("vector_store_ids", []) + point_ids,
        "should_scrape": False,
        "messages": state["messages"] + [AIMessage(content=f"I've loaded the blog '{blog_title}'. What would you like to know about it?")],
        "step_count": state['step_count'] + 1
    })
