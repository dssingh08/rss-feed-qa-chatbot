""" Summarizer Node - Summarize conversation history """

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.models import get_internal_model
from src.utils.prompts import MEMORY_SUMMARIZATION_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def summarizer_node(state: AgentState) -> AgentState:
    """
    Summarize the conversation history and store it in the agent state.
    """
    logger.info("=== SUMMARIZER NODE ===")

    conversation_history = state.get("messages", [])
    
    formatted_conversation = []
    for msg in conversation_history:
        if isinstance(msg, HumanMessage):
            formatted_conversation.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_conversation.append(f"Agent: {msg.content}")
    
    conversation_text = "\n".join(formatted_conversation)

    if not conversation_text.strip():
        logger.info("No conversation history to summarize.")
        return state.copy()

    model_choice = state.get("response_model", "gemini")
    model = get_internal_model(model_choice)
    
    response = await model.ainvoke([HumanMessage(content=MEMORY_SUMMARIZATION_PROMPT.format(conversation=conversation_text))])
    summary = response.content.strip()

    logger.info(f"Generated conversation summary: {summary}")

    new_state = state.copy()
    new_state.update({
        "conversation_summary": summary,
        "step_count": state["step_count"] + 1
    })
    return new_state
