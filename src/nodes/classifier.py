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

logger = setup_logger(__name__)


class QueryClassification(BaseModel):
    query_type: Literal["discovery", "direct", "general", "blog_selection", "unknown"] = Field(
        ..., description="Classification of the user's query"
    )
    company_name: Optional[str] = Field(
        None, description="The name of the company mentioned in the query, or null if not specified"
    )
    topic: Optional[str] = Field(
        None, description="The specific topic of interest for direct queries, or null for discovery queries"
    )
    reasoning: str = Field(..., description="A brief explanation for the classification")


async def classify_query_node(state: AgentState) -> AgentState:
    """
    Classify user query into discovery, direct, general, or blog_selection type
    """
    logger.info("=== CLASSIFY QUERY NODE ===")

    last_message = state["messages"][-1].content
    blog_titles = state.get("blog_titles", []) # Get blog titles from state
    logger.debug(f"classifying query: {last_message}, with blog_titles present: {bool(blog_titles)}")

    model = get_internal_model()

    # Adjust prompt based on whether blog titles were previously listed
    if blog_titles:
        titles_list = "\n".join([f"- {blog['title']}" for blog in blog_titles])
        system_prompt_content = QUERY_CLASSIFIER_PROMPT + (
            f"\n\nNote: A list of blog titles was previously presented to the user:\n{titles_list}\n"
            "If the user's query is a selection from this list (by number or title), classify it as 'blog_selection'."
        )
    else:
        system_prompt_content = QUERY_CLASSIFIER_PROMPT

    messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=f"User query: {last_message}")
    ]

    classification_result = None
    for i in range(3):  # Retry up to 3 times
        try:
            response = await model.ainvoke(messages)
            logger.debug(f"Classifier response attempt {i+1}: {response.content}")
            
            # Attempt to parse with Pydantic
            # Extract JSON from markdown code block if present
            json_string = str(response.content).strip()
            if json_string.startswith("```json") and json_string.endswith("```"):
                json_string = json_string[len("```json"): -len("```")].strip()
            
            classification_result = QueryClassification.model_validate_json(json_string)
            break
        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Attempt {i+1} failed to parse or validate classification response: {e}")
            if i == 2:
                logger.error(f"Failed to parse or validate classification response after multiple retries: {e}")
                # Fallback to default unknown classification
                classification_result = QueryClassification(
                    query_type="unknown",
                    company_name=None,
                    topic=None,
                    reasoning="Failed to classify after multiple retries."
                )
            await asyncio.sleep(1) # Optionally, add a delay before retrying

    if classification_result is None:
        # This case should ideally be caught by the last retry's fallback, but as a safeguard
        classification_result = QueryClassification(
            query_type="unknown",
            company_name=None,
            topic=None,
            reasoning="Classification failed unexpectedly."
        )

    logger.info(f"Classification: type= {classification_result.query_type}, company={classification_result.company_name}, topic={classification_result.topic}")

    new_state = state.copy()
    new_state.update({
        "query_type": classification_result.query_type,
        "company_name": classification_result.company_name,
        "topic": classification_result.topic,
        "selected_blog_index": getattr(classification_result, "selected_blog_index", None), # Safely get attribute
        "selected_blog_title": getattr(classification_result, "selected_blog_title", None), # Safely get attribute
        "step_count": state["step_count"] + 1
    })
    return new_state
