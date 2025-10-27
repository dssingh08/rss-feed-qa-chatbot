""" RSS Selection Processor Node - Handles user selection of RSS feed entry """

from langchain_core.messages import AIMessage
from src.state import AgentState
from src.utils.logger import setup_logger
from src.tools.vector_operations import vector_store # Import vector_store

logger = setup_logger(__name__)


async def rss_selection_processor_node(state: AgentState) -> AgentState:
    """
    Processes user's selection from RSS feed titles and updates state.
    """
    logger.info("=== RSS SELECTION PROCESSOR NODE ===")

    blog_titles = state.get("blog_titles", [])
    new_state = state.copy()

    if not blog_titles:
        logger.warning("No blog titles found in state for selection processing.")
        new_state.update({
            "should_scrape": False,
            "messages": state["messages"] + [AIMessage(content="I couldn't find any blogs to select from.")],
            "step_count": state["step_count"] + 1
        })
        return new_state

    query_type = state.get("query_type")
    selected_blog_index = state.get("selected_blog_index")
    selected_blog_title_from_classifier = state.get("selected_blog_title")

    selected_blog = None

    if query_type == "blog_selection":
        if selected_blog_index is not None and 1 <= selected_blog_index <= len(blog_titles):
            selected_blog = blog_titles[selected_blog_index - 1]
        elif selected_blog_title_from_classifier:
            for blog in blog_titles:
                if selected_blog_title_from_classifier.lower() in blog['title'].lower():
                    selected_blog = blog
                    break
    
    if selected_blog:
        logger.info(f"User selected blog: {selected_blog['title']}")
        
        if vector_store.check_blog_exists(selected_blog["link"]):
            should_scrape = False
            scraper_reason = f"Blog content for '{selected_blog['title']}' already exists in vector store."
            logger.info(scraper_reason)
        else:
            should_scrape = True
            scraper_reason = f"User selected blog: {selected_blog['title']}"

        new_state.update({
            "selected_blog_url": selected_blog["link"],
            "selected_blog_title": selected_blog["title"],
            "should_scrape": should_scrape,
            "query_type": "direct",
            "scraper_reason": scraper_reason,
        })
        logger.info(f"Updated state with selected blog: {selected_blog['link']}")
    else:
        logger.info("No valid blog selection found.")
        new_state.update({
            "should_scrape": False,
            "messages": state["messages"] + [AIMessage(content="I didn't understand your selection. Please try again or choose another option.")],
        })
    
    new_state.update({"step_count": state["step_count"] + 1})
    return new_state
