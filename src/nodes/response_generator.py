""" Response Generator Node """
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from src.models import get_response_model
from src.tools.vector_operations import vector_store
from src.utils.prompts import RESPONSE_GENERATION_PROMPT, GENERAL_RESPONSE_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def generate_response_node(state: AgentState) -> AgentState:
    """
    Generate response based on retrieved context
    """
    logger.info("=== GENERATE RESPONSE NODE ===")

    user_query = str(state["messages"][-1].content)
    company_name = state.get("company_name")
    model_choice = state.get("response_model", "gemini")
    query_type = state.get("query_type", "unknown")

    logger.info(f"Generating response with model: {model_choice}, query_type: {query_type}")

    model = get_response_model(model_choice)
    response_text = ""
    blog_titles = state.get("blog_titles", [])

    # Only list blogs if the query type was 'discovery' (initial request for blogs)
    if blog_titles and query_type == "discovery":
        logger.info("Generating response with blog titles list")
        titles_text = "\n".join([f"{idx}. {blog['title']}" for idx, blog in enumerate(blog_titles, 1)])
        response_text = f"Here are the latest blog posts I found:\n{titles_text}\n\nPlease tell me which one you're interested in (e.g., '1' or 'Tell me about \"{blog_titles[0]['title']}\"')."
        new_state = state.copy()
        new_state.update({
            "messages": state["messages"] + [AIMessage(content=response_text)],
            "step_count": state["step_count"] + 1
        })
        return new_state
    elif query_type == "general":
        logger.info("Generating general response")
        messages = state["messages"][:-1] + [
            SystemMessage(content="You are a helpful and friendly AI assistant."),
            HumanMessage(content=user_query)
        ]
        response = await model.ainvoke(messages)
        response_text = response.content
    else:
        results = vector_store.search_blog_content(query=user_query,
                                                   company_name=company_name,
                                                   limit=10)
        
        if not results:
            logger.warning("No context found in vector store for discovery/direct query")
            response_text = "I couldn't find relevant blog content for that query. Please try a different topic or company."
        else:
            context = "\n\n".join([
                f"From '{r['blog_title']}':\n{r['content']}"
                for r in results
            ])

            logger.debug(f"Retrieved {len(results)} context chunks")

            messages = state["messages"][:-1] + [
                SystemMessage(content=f"You are a helpful AI assistant explaining blog content. Use the following context:\n\n{context}"),
                HumanMessage(content=user_query)
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
