""" Main LangGraph Agent """

import platform
import asyncio
from typing import Literal, Union
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState, QueryType 
from src.nodes.classifier import classify_query_node
from src.nodes.rss_fetcher import fetch_rss_titles_node
from src.nodes.blog_search import search_blog_node
from src.nodes.scraper import scraper_node
from src.nodes.response_generator import generate_response_node
from src.nodes.rss_selection_processor import rss_selection_processor_node
from src.nodes.summarizer import summarizer_node
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def route_after_classify(state: AgentState) -> str:
    """ Route based on query classification """
    query_type = state.get("query_type", "unknown")
    blog_titles = state.get("blog_titles", [])
    logger.info(f"Routing after classify: {query_type}, blog_titles present: {bool(blog_titles)}")

    # If it's a blog selection query but no blog titles have been fetched yet,
    # route to fetch RSS first to populate blog_titles.
    if query_type == "blog_selection" and not blog_titles:
        logger.info("Blog selection query with no existing blog titles, routing to Discover Blogs.")
        return "discovery_for_selection" # A temporary route to fetch RSS
    
    return query_type


def route_check_scrape(state: AgentState) -> Literal["Scrape Blog Content", "Generate Response", END]:
    """ Universal router for endpoints that supply a selected URL to scrape """
    should_scrape = state.get("should_scrape", False)
    selected_blog_url = state.get("selected_blog_url")
    logger.info(f"Checking scrape routing: should_scrape={should_scrape}, url={selected_blog_url}")

    if should_scrape and selected_blog_url:
        return "Scrape Blog Content"
    else:
        return "Generate Response"


def route_after_rss_fetch(state: AgentState) -> Literal["Process Blog Selection", "Generate Response", END]:
    """ Route after RSS fetch - if titles are present, go to process or generate response. Else END """
    blog_titles = state.get("blog_titles", [])
    query_type = state.get("query_type", "unknown")
    
    if blog_titles:
        if query_type == "blog_selection":
            return "Process Blog Selection"
        return "Generate Response"
    else:
        return END


def route_after_generation(state: AgentState) -> Literal["Summarize Conversation", END]:
    """ Route after generate response based on step count """
    step_count = state.get("step_count", 0)
    logger.info(f"Routing after generation: step_count={step_count}")

    if step_count > 0 and step_count % 3 == 0:  
        return "Summarize Conversation"
    return END


def create_graph():
    """ Create the LangGraph workflow """
    logger.info("Building LangGraph workflow")

    workflow = StateGraph(AgentState)

    workflow.add_node("Classify Query", classify_query_node)
    workflow.add_node("Discover Blogs", fetch_rss_titles_node)
    workflow.add_node("Process Blog Selection", rss_selection_processor_node)
    workflow.add_node("Find Blog by Topic", search_blog_node)
    workflow.add_node("Scrape Blog Content", scraper_node)
    workflow.add_node("Generate Response", generate_response_node)
    workflow.add_node("Summarize Conversation", summarizer_node) 


    workflow.set_entry_point("Classify Query")

    workflow.add_conditional_edges(
        "Classify Query",
        route_after_classify,
        {
            "discovery": "Discover Blogs",
            "direct": "Find Blog by Topic",
            "blog_selection": "Process Blog Selection",
            "contextual_qa": "Generate Response",
            "general": "Generate Response",
            "unknown": "Generate Response",
            "discovery_for_selection": "Discover Blogs" 
        }
    )

    workflow.add_conditional_edges(
        "Find Blog by Topic",
        route_check_scrape,
        {
            "Scrape Blog Content": "Scrape Blog Content",
            "Generate Response": "Generate Response",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "Discover Blogs",
        route_after_rss_fetch,
        {
            "Process Blog Selection": "Process Blog Selection",
            "Generate Response": "Generate Response",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "Process Blog Selection",
        route_check_scrape,
        {
            "Scrape Blog Content": "Scrape Blog Content",
            "Generate Response": "Generate Response",
            END: END
        }
    )

    workflow.add_edge("Scrape Blog Content", "Generate Response")
    
    workflow.add_conditional_edges(
        "Generate Response",
        route_after_generation, 
        {
            "Summarize Conversation": "Summarize Conversation", 
            END: END
        }
    )

    workflow.add_edge("Summarize Conversation", END)

    checkpointer = MemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    logger.info("LangGraph workflow built successfully")

    return graph


graph = create_graph()

async def invoke_graph(state: AgentState):
    try:
        return await graph.ainvoke(state)
    except Exception as e:
        logger.error(f"Error during graph invocation: {e}", exc_info=True)
        raise
