""" Blog Search Node - Find specific blog for Type2 """

from langchain_core.messages import HumanMessage, AIMessage
from src.state import AgentState
from src.models import get_internal_model
from src.tools.rss_parser import RSSParser, get_company_feed
from src.utils.prompts import BLOG_SEARCH_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


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
            parser = RSSParser()
            found_blog = parser.search_feed(feed_url, search_query)

            if found_blog:
                logger.info(f"Found matching blog: {found_blog['title']}")

                new_state.update({
                    "selected_blog_url": found_blog["link"],
                    "selected_blog_title": found_blog["title"],
                    "should_scrape": True,
                    "scraper_reason": f"User wants to learn about '{topic}' from {company_name}",
                    "step_count": state["step_count"] + 1
                })
                return new_state
    logger.warning("No omatching blog found")
    new_state.update({
        "messages": state["messages"] + [AIMessage(content=f"I couldn't find a specific blog about '{topic}' from {company_name}. Could you rephrase or try a different topic?")],
        "step_count": state["step_count"] + 1
    })
    return new_state