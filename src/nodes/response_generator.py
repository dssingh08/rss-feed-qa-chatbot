""" Response Generator Node """
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.models import get_response_model
from src.tools.vector_operations import vector_store
from src.utils.prompts import RESPONSE_GENERATION_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def generate_response_node(state: AgentState) -> AgentState:
    """
    Generate response based on retrieved context
    """
    logger.info("=== GENERATE RESPONSE NODE ===")

    user_query = state["messages"][-1].content
    company_name = state.get("company_name")
    model_choice = state.get("response_model", "gemini")

    logger.info(f"Generating response with model: {model_choice}")

    results = vector_store.search_blog_content(query=user_query,
                                               company_name=company_name,
                                               limit=5)
    
    if not results:
        logger.warning("No context found in vector store")

        response_text = "I don't have enough information to answer that question. Could you select a blog first or ask about a different topic?"
    else:
        context = "\n\n".join([
            f"From '{r['blog_title']}':\n{r['content']}"
            for r in results
        ])

        logger.debug(f"Retrieved {len(results)} context chunks")

        model = get_response_model(model_choice)

        prompt = RESPONSE_GENERATION_PROMPT.format(
            context=context,
            question=user_query
        )
        messages = [
            SystemMessage(content="You are a helpfull AI assistant explaining blog content."),
            HumanMessage(content=prompt)
        ]

        response = await model.ainvoke(messages)
        response_text = response.content

        sources = list(set([r['blog_title'] for r in results]))
        response_text += f"\n\n*Source: {','.join(sources[:2])}*"

    logger.info("Response generated successfully")
    new_state = state.copy()
    new_state.update({
        "messages": state["messages"] + [AIMessage(content=response_text)],
        "step_count": state["step_count"] + 1
    })

    return new_state