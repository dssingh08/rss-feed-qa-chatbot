""" Blog Search Node - Find specific blog for Type2 """

from langchain_core.messages import HumanMessage, AIMessage
from src.state import AgentState
from src.models import get_internal_model
from src.tools.rss_parser import RSSParser, get_company_feed
from src.utils.prompts import BLOG_SEARCH_PROMPT, LLM_BLOG_SELECTION_PROMPT
from src.utils.logger import setup_logger
import re # Import re at the top

logger = setup_logger(__name__)


async def llm_select_best_blog(user_query: str, blog_entries: list) -> dict:
    """
    Ask an LLM to select the best blog post from RSS entries based on user query
    
    Args:
        user_query: the original user question or intent
        blog_entries: List of dicts with blog info (title, description, link)
    
    Returns:
        The dict representing the chosen blog or None
    """
    if not blog_entries:
        return None
    
    entries_text = "\n".join(
        [f"{i+1}. Title: {b['title']}\n   Description: {b['description'][:200]}" for i, b in enumerate(blog_entries)]
    )
        
    model = get_internal_model()
    
    response = await model.ainvoke([HumanMessage(content=LLM_BLOG_SELECTION_PROMPT.format(
        user_query=user_query,
        entries_text=entries_text,
        n = len(blog_entries)
    ))])
    logger.info(f"LLM blog selection response: {response.content}")
    
    try:
        # Use regex to find the first number in the response
        match = re.search(r'\d+', response.content.strip())
        if match:
            choice_num = int(match.group(0))
            if 1 <= choice_num <= len(blog_entries):
                return blog_entries[choice_num - 1]
            elif choice_num == 0: # LLM explicitly said no relevant blog
                logger.info("LLM indicated no relevant blog found.")
                return None
        logger.warning(f"Could not extract a valid blog selection number from LLM response: {response.content}")
    except Exception as e:
        logger.error(f"Failed to parse LLM blog selection: {e}")
    
    return None 


async def search_blog_node(state: AgentState) -> AgentState:
    """
    Search for specific blog post based on topic
    """
    logger.info("=== SEARCH BLOG NODE ===")

    company_name = state.get("company_name")
    topic = state.get("topic")
    user_query = state["messages"][-1].content

    new_state = state.copy()

    logger.info(f"Searching for blog: company={company_name}, topic={topic}")

    model = get_internal_model()
    search_prompt = BLOG_SEARCH_PROMPT.format(
        user_query=user_query,
        company_name=company_name or "Any"
    )

    response = await model.ainvoke([HumanMessage(content=search_prompt)])
    search_query = response.content.strip()
    logger.info(f"Generated search query: {search_query}")

    if company_name:
        feed_url = get_company_feed(company_name)
        if feed_url:
            rss_entries = RSSParser.parse_feed(feed_url, max_entries=10)
            found_blog = await llm_select_best_blog(user_query, rss_entries)

            if found_blog:
                logger.info(f"Found matching blog: {found_blog['title']}")

                new_state.update({
                    "selected_blog_url": found_blog["link"],
                    "selected_blog_title": found_blog["title"],
                    "should_scrape": True,
                    "scraper_reason": f"LLM selected this blog as most relevant",
                    "step_count": state["step_count"] + 1
                })
                return new_state
    logger.warning("No matching blog found")
    new_state.update({
        "selected_blog_url": None,
        "selected_blog_title": None,
        "should_scrape": False,
        "messages": state["messages"] + [AIMessage(content=f"I couldn't find a specific blog about '{topic}' from {company_name}. Could you rephrase or try a different topic?")],
        "step_count": state["step_count"] + 1
    })
    return new_state
