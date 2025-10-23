""" Main LangGraph Agent """

import platform
import asyncio
from typing import Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState
from src.nodes.classifier import classify_query_node
from src.nodes.rss_fetcher import fetch_rss_titles_node
from src.nodes.blog_search import search_blog_node
from src.nodes.scraper import scraper_node
from src.nodes.response_generator import generate_response_node
from src.nodes.rss_selection_processor import rss_selection_processor_node # New import
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
    elif query_type == "general": 
        return "generate_response"
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


def route_after_rss_fetch(state: AgentState) -> Literal["generate_response", END]:
    """ Route after RSS fetch - if titles are present, go to generate response, else END """
    blog_titles = state.get("blog_titles", [])
    if blog_titles:
        return "generate_response"
    else:
        return END

def route_after_rss_selection(state: AgentState) -> Literal["scraper", "generate_response", END]:
    """ Route after RSS selection processing """
    should_scrape = state.get("should_scrape", False)
    selected_blog_url = state.get("selected_blog_url")
    logger.info(f"Routing after RSS selection: should_scrape={should_scrape}, url_present={bool(selected_blog_url)}")

    if should_scrape and selected_blog_url:
        return "scraper"
    else:
        return "generate_response" # If no blog was selected or scraping is not needed, go to generate response


def route_after_generate_response(state: AgentState) -> Literal["rss_selection_processor", END]:
    """ Route after generate response based on query type """
    query_type = state.get("query_type", "unknown")
    logger.info(f"Routing after generate response: {query_type}")
    if query_type == "blog_selection":
        return "rss_selection_processor"
    return END


def create_graph():
    """ Create the LangGraph workflow """
    logger.info("Building LangGraph workflow")

    workflow = StateGraph(AgentState)

    workflow.add_node("classify_query", classify_query_node)
    workflow.add_node("fetch_rss", fetch_rss_titles_node)
    workflow.add_node("rss_selection_processor", rss_selection_processor_node) # New node
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

    # New conditional edge after fetch_rss to go to the response generator
    workflow.add_conditional_edges(
        "fetch_rss",
        route_after_rss_fetch,
        {
            "generate_response": "generate_response",
            END: END
        }
    )

    # Conditional edge after the new selection processor node
    workflow.add_conditional_edges(
        "rss_selection_processor",
        route_after_rss_selection,
        {
            "scraper": "scraper",
            "generate_response": "generate_response",
            END: END # Added END for completeness, though generate_response might cover it
        }
    )

    workflow.add_edge("scraper", "generate_response")
    
    # New conditional edge after generate_response
    workflow.add_conditional_edges(
        "generate_response",
        route_after_generate_response,
        {
            "rss_selection_processor": "rss_selection_processor",
            END: END
        }
    )

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("LangGraph workflow built successfully")

    return graph


graph = create_graph()
