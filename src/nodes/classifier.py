""" Query Classifier Node """

import json
import asyncio
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.models import get_internal_model
from src.utils.prompts import QUERY_CLASSIFIER_PROMPT
from src.utils.logger import setup_logger
from src.config import settings # Import settings

logger = setup_logger(__name__)


class QueryClassification(BaseModel):
    query_type: Literal["discovery", "direct", "general", "blog_selection", "contextual_qa", "unknown"] = Field(
        ..., description="Classification of the user's query"
    )
    company_name: Optional[str] = Field(
        None, description="The name of the company mentioned in the query, or null if not specified"
    )
    topic: Optional[str] = Field(
        None, description="The specific topic of interest for direct queries, or null for discovery queries"
    )
    selected_blog_index: Optional[int] = Field(
        None, description="The 1-based index of the selected blog post if query_type is 'blog_selection'"
    )
    selected_blog_title: Optional[str] = Field(
        None, description="The title of the selected blog post if query_type is 'blog_selection'"
    )
    reasoning: str = Field(..., description="A brief explanation for the classification")


async def classify_query_node(state: AgentState) -> AgentState:
    """
    Classify user query into discovery, direct, general, or blog_selection type
    """
    logger.info("=== CLASSIFY QUERY NODE ===")

    last_message = state["messages"][-1].content
    blog_titles = state.get("blog_titles", [])
    conversation_summary = state.get("conversation_summary")
    selected_blog_title = state.get("selected_blog_title")
    selected_blog_url = state.get("selected_blog_url")

    logger.debug(f"classifying query: {last_message}, with blog_titles present: {bool(blog_titles)}, conversation_summary present: {bool(conversation_summary)}, selected_blog_title present: {bool(selected_blog_title)}")

    model = get_internal_model()

    conversation_summary_context = ""
    if conversation_summary:
        conversation_summary_context = f"Current conversation summary: {conversation_summary}\n\n"

    supported_companies_context = f"Supported companies: {', '.join(settings.supported_companies)}\n\n"

    active_blog_context = ""
    if selected_blog_title and selected_blog_url:
        active_blog_context = (
            f"The user is currently discussing the blog post: '{selected_blog_title}' "
            f"(URL: {selected_blog_url}).\n\n"
        )

    system_prompt_content = QUERY_CLASSIFIER_PROMPT.format(
        conversation_summary_context=conversation_summary_context,
        supported_companies_context=supported_companies_context,
        active_blog_context=active_blog_context
    )

    if blog_titles:
        titles_list = "\n".join([f"- {blog['title']}" for blog in blog_titles])
        system_prompt_content += (
            f"\n\nNote: A list of blog titles was previously presented to the user:\n{titles_list}\n"
            "If the user's query is a selection from this list (by number or title), classify it as 'blog_selection'."
        )

    messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=f"User query: {last_message}")
    ]

    classification_result = None
    for i in range(3):
        try:
            response = await model.ainvoke(messages)
            logger.debug(f"Classifier response attempt {i+1}: {response.content}")
            
            json_string = str(response.content).strip()
            if json_string.startswith("```json") and json_string.endswith("```"):
                json_string = json_string[len("```json"): -len("```")].strip()
            
            classification_result = QueryClassification.model_validate_json(json_string)
            break
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Attempt {i+1} failed to parse or validate classification response: {e}")
            if i == 2:
                logger.error(f"Failed to parse or validate classification response after multiple retries: {e}")
                classification_result = QueryClassification(
                    query_type="unknown",
                    company_name=None,
                    topic=None,
                    selected_blog_index=None,
                    selected_blog_title=None,
                    reasoning="Failed to classify after multiple retries."
                )
            await asyncio.sleep(1)

    if classification_result is None:
        classification_result = QueryClassification(
            query_type="unknown",
            company_name=None,
            topic=None,
            selected_blog_index=None,
            selected_blog_title=None,
            reasoning="Classification failed unexpectedly."
        )

    logger.info(f"Classification: type= {classification_result.query_type}, company={classification_result.company_name}, topic={classification_result.topic}, index={classification_result.selected_blog_index}, title='{classification_result.selected_blog_title}'")

    if classification_result.query_type == "blog_selection":
        if classification_result.selected_blog_index is None and classification_result.selected_blog_title is None:
            logger.warning("Query classified as 'blog_selection' but no index or title was extracted. Reclassifying as 'general'.")
            classification_result.query_type = "general"
            classification_result.reasoning = "Reclassified: blog_selection query without specific index or title."

    new_state = state.copy()
    new_state.update({
        "query_type": classification_result.query_type,
        "company_name": classification_result.company_name,
        "topic": classification_result.topic,
        "selected_blog_index": classification_result.selected_blog_index,
        "selected_blog_title": classification_result.selected_blog_title,
        "step_count": state["step_count"] + 1
    })
    return new_state
