""" Main LangGraph Agent """

from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState
from src.nodes.classifier import classify_query_node
from src.nodes.rss_fetcher import fetch_rss_titles_node
from src.nodes.blog_search import search_blog_node
from src.nodes.scraper import scraper_node
from src.nodes.response_generator import generate_response_node
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def route_after_classify(state: AgentState) -> Literal["fetch_rss", "search_blog", "generate_response"]:
    """ Route based on query classification """
    query_type = state.get("query_type", "unknown")
    logger.info(f"Routing after classify: {query_type}")

    if query_type == "discovery":
        return "fetch_rss"
    elif query_type == "direct":
        return "search_blog"
    else:
        return "generate_response"


def route_after_search(state: AgentState) -> Literal["scraper", "generate_response"]:
    """ Route after blog search """
    should_scrape = state.get("should_scrape", False)
    logger.info(f"Routing after search: scrape={should_scrape}")

    if should_scrape and state.get("selected_blog_url"):
        return "scraper"
    else:
        return "generate_response"


def route_after_rss(state: AgentState) -> Literal["scraper", "generate_response", END]:
    """ Route after RSS fetch - wait for user selection """
    blog_titles = state.get("blog_titles", [])

    if not blog_titles:
        return END

    last_message = state["messages"][-1].content.lower() if state['messages'] else ""
    
    for idx, blog in enumerate(blog_titles, 1):
        if str(idx) in last_message or blog['title'].lower() in last_message:
            logger.info(f"User selected blog: {blog['title']}")
            # Update state with selection
            state["selected_blog_url"] = blog["link"]
            state["selected_blog_title"] = blog["title"]
            state["should_scrape"] = True
            state["scraper_reason"] = f"User selected blog: {blog['title']}"
            return "scraper"
        
    return END


def create_graph():
    """ Create the LangGraph workflow """
    logger.info("Building LangGraph workflow")

    workflow = StateGraph(AgentState)

    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("fetch_rss", fetch_rss_titles_node)
    workflow.add_node("search_blog", search_blog_node)
    workflow.add_node("scraper", scraper_node)
    workflow.add_node("generate_response", generate_response_node)


    workflow.set_entry_point("classify_query")

    workflow.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "fetch_rss": "fetch_rss",
            "search_blog": "search_blog",
            "generate_response": "generate_response"
        }
    )

    workflow.add_conditional_edges(
        "search_blog",
        route_after_search,
        {
            "scraper": "scraper",
            "generate_response": "generate_response"
        }
    )


    workflow.add_conditional_edges(
        "fetch_rss",
        route_after_rss,
        {
            "scraper": "scraper",
            "generate_response": "generate_response",
            END: END
        }
    )

    workflow.add_edge("scraper", "generate_response")
    workflow.add_edge("generate_response", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("LangGraph workflow built successfully")

    return graph


graph = create_graph()