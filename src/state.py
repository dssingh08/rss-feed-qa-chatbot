""" Agent State Definitions """

from typing import Annotated, Optional, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """ State for the RSS Q&A Agent """

    messages: Annotated[list[BaseMessage], add_messages]

    summary: str

    user_id: str

    query_type: Literal["discovery", "direct", "general", "blog_selection", "unknown"]

    company_name: Optional[str]
    topic: Optional[str]
    selected_blog_index: Optional[int]

    blog_titles: Optional[list[dict]]

    selected_blog_url: Optional[str]
    selected_blog_title: Optional[str]

    vector_store_ids: list[str]

    should_scrape: bool
    scraper_reason: Optional[str]

    response_model: str

    step_count: int
