import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from src.tools.evaluation_tool import DeepEvalOpenRouterJudge
from src.models import get_eval_judge
import asyncio

# The PyTest-Asyncio plugin requires this to run async tests
pytest_plugins = ('pytest_asyncio',)

def get_judge():
    model = get_eval_judge()
    return DeepEvalOpenRouterJudge(model)

@pytest.mark.asyncio
async def test_rag_quality_smoke():
    """
    Smoke test for the RAG evaluation. 
    This checks if the DeepEval judge works with our OpenRouter wrapper.
    """
    input_query = "What is the capital of France?"
    retrieval_context = ["Paris is the capital and most populous city of France."]
    actual_output = "Paris"

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    judge = get_judge()
    
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge),
        AnswerRelevancyMetric(threshold=0.7, model=judge)
    ]

    assert_test(test_case, metrics)

@pytest.mark.asyncio
async def test_rag_hallucination_catch():
    """
    Test that the FaithfulnessMetric successfully CATCHES and FAILS a hallucinated answer.
    """
    input_query = "What features does the new product have?"
    retrieval_context = ["The new product features an OLED screen and a titanium body."]
    # AI hallucinates a battery life feature
    actual_output = "The new product features an OLED screen, a titanium body, and a 50-hour battery life."

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    judge = get_judge()
    metric = FaithfulnessMetric(threshold=0.7, model=judge)
    
    await metric.a_measure(test_case)
    # The metric should evaluate to FALSE (failed) because of the hallucination
    assert not metric.is_successful(), f"Faithfulness metric failed to catch the hallucination. Reason: {metric.reason}"


@pytest.mark.asyncio
async def test_rag_answer_relevancy_catch():
    """
    Test that the AnswerRelevancyMetric successfully CATCHES an off-topic/evasive answer.
    """
    input_query = "Who is the CEO of Google?"
    retrieval_context = ["Sundar Pichai is the chief executive officer of Alphabet Inc. and its subsidiary Google."]
    actual_output = "Google is a large tech company that focuses on search engine technology."

    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output,
        retrieval_context=retrieval_context
    )

    judge = get_judge()
    metric = AnswerRelevancyMetric(threshold=0.7, model=judge)
    
    await metric.a_measure(test_case)
    # The metric should evaluate to FALSE (failed) because it didn't answer the question
    assert not metric.is_successful(), f"Answer Relevancy metric failed to catch the evasion. Reason: {metric.reason}"


@pytest.mark.asyncio
async def test_router_accuracy():
    """ 
    Integration Test for the Classifier Node.
    Ensures that a discovery query is properly parsed as 'discovery' with the correct company.
    """
    from src.nodes.classifier import query_classifier_node
    from langchain_core.messages import HumanMessage
    
    # Mock agent state
    state = {
        "messages": [HumanMessage(content="Show me the latest blogs from Anthropic please.")],
        "conversation_summary": "",
        "company_name": None,
        "query_type": "unknown",
        "step_count": 0
    }
    
    # Run the real LangGraph node
    new_state = await query_classifier_node(state)
    
    assert new_state["query_type"] == "discovery", f"Expected discovery, got {new_state['query_type']}"
    assert new_state["company_name"].lower() == "anthropic", f"Expected anthropic, got {new_state['company_name']}"


@pytest.mark.asyncio
async def test_tool_calling_integrity():
    """ 
    Unit Test for Tool Calling integrity of the response generator prompt.
    Ensures that when the context contains a hyperlink and the user asks about it, 
    the scraper decision properly triggers.
    """
    from src.nodes.rss_selection_processor import rss_selection_processor_node
    
    # Mock an AgentState where the user has selected a blog
    state = {
        "messages": [HumanMessage(content="I'll take the first one.")],
        "blog_titles": [{"title": "Claude 3.5 Sonnet Released", "link": "https://anthropic.com/claude3-5"}],
        "selected_blog_index": 1,
        "query_type": "blog_selection",
        "step_count": 0
    }
    
    new_state = await rss_selection_processor_node(state)
    
    assert new_state["selected_blog_url"] == "https://anthropic.com/claude3-5"
    assert new_state["should_scrape"] is True, "The node should have decided to scrape the unfamiliar blog URL."
