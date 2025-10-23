""" RSS Fetcher Node - Fetch blog titles for type 1 queries """

from langchain_core.messages import HumanMessage, AIMessage
from src.state import AgentState
from src.tools.rss_parser import RSSParser, get_company_feed
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def fetch_rss_titles_node(state: AgentState) -> AgentState:
    """
    Fetch recent blog titles from company RSS feed
    """

    logger.info("=== FETCH RSS TITLES NODE ===")
    company_name = state.get("company_name")
    new_state = state.copy()
    if not company_name:
        logger.warning("No company name provided")
        new_state.update({
            "blog_titles": [],
            "messages": state["messages"] + [
                AIMessage(content="I couldn't identify which company you're asking about. Could you please specify?")
            ],
            "step_count": state["step_count"] + 1
        })
        return new_state

    feed_url = get_company_feed(company_name)
    if not feed_url:
        logger.warning(f"No RSS feed found for {company_name}")
        new_state.update({
            "blog_titles": [],
            "messages": state["messages"] + [
                AIMessage(content=f"Sorry, I don't have an RSS feed configured for {company_name}.")
            ],
            "step_count": state["step_count"] + 1
        })
        return new_state
    
    parser = RSSParser()
    blog_titles = parser.parse_feed(feed_url, max_entries=settings.max_blog_titles)

    logger.info(f"Fetched {len(blog_titles)} blog titles from {company_name}")

    if blog_titles:
        response_text = f"Here are the latest blogs from {company_name}:\n\n"

        for idx, blog in enumerate(blog_titles, 1):
            response_text += f"{idx}. **{blog['title']}**\n"
            response_text += f"{blog['published']}\n"
            response_text += f"{blog['description'][:150]}...\n\n"
        response_text += "Which one would you like to learn more about?"
    else:
        response_text = f"I couldn't find any recent blogs from {company_name}."
    
    new_state.update({
        "blog_titles": blog_titles,
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "step_count": state["step_count"] + 1
    })

    new_state.update({"should_scrape": False})

    return new_state
