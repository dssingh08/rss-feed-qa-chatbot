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

    user_query = str(state["messages"][-1].content) if state["messages"] and state["messages"][-1].content else ""
    company_name = state.get("company_name")
    model_choice = state.get("response_model", "gemini")
    query_type = state.get("query_type", "unknown")

    logger.info(f"Generating response with model: {model_choice}, query_type: {query_type}")

    model = get_response_model(model_choice)
    response_text = ""
    blog_titles = state.get("blog_titles", [])
    unique_sources = {} 

    try:
        if blog_titles and query_type == "discovery":
            logger.info("Generating response with blog titles list")
            titles_text = "\n".join([f"{idx}. {blog['title']}" for idx, blog in enumerate(blog_titles, 1)])
            response_text = f"Here are the latest blog posts I found:\n{titles_text}\n\nPlease tell me which one you're interested in (e.g., '1' or 'Tell me about \"{blog_titles[0]['title']}\"')."
        elif query_type == "general":
            logger.info("Generating general response")
            conversation_summary = state.get("conversation_summary")
            selected_blog_title = state.get("selected_blog_title")
            selected_blog_url = state.get("selected_blog_url")

            context_for_general_query = ""
            if conversation_summary:
                context_for_general_query += f"Conversation Summary: {conversation_summary}\n\n"
            if selected_blog_title and selected_blog_url:
                context_for_general_query += f"Currently discussing: '{selected_blog_title}' (URL: {selected_blog_url})\n\n"

            if context_for_general_query:
                logger.debug(f"Context for general query: {context_for_general_query}")
                messages = [
                    SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(context=context_for_general_query, question=user_query)),
                    HumanMessage(content=user_query)
                ]
            else:
                logger.debug("No specific context for general query, using basic prompt.")
                messages = [
                    SystemMessage(content="You are a helpful and friendly AI assistant."),
                    HumanMessage(content=user_query)
                ]
            
            logger.debug(f"Messages for general query: {messages}")
            response = await model.ainvoke(messages)
            logger.debug(f"Raw response for general query: {response}")
            response_text = response.content
        elif query_type == "contextual_qa":
            logger.info("Generating contextual Q&A response")
            search_query_for_vector_store = user_query
            blog_url_filter = state.get("selected_blog_url")
            company_name_filter = state.get("company_name")

            if not blog_url_filter:
                logger.warning("Contextual QA query but no selected_blog_url found. Falling back to general response.")
                messages = [
                    SystemMessage(content="You are a helpful and friendly AI assistant."),
                    HumanMessage(content=user_query)
                ]
                response = await model.ainvoke(messages)
                response_text = response.content
            else:
                search_query_for_vector_store = str(search_query_for_vector_store)

                results = vector_store.search_blog_content(query=search_query_for_vector_store,
                                                           company_name=company_name_filter,
                                                           blog_url=blog_url_filter,
                                                           limit=100)

                if not results:
                    logger.warning(f"No context found in vector store for contextual QA query '{user_query}' with URL '{blog_url_filter}'. Falling back to general knowledge.")
                    
                    conversation_summary = state.get("conversation_summary")
                    selected_blog_title = state.get("selected_blog_title")
                    
                    fallback_context = ""
                    if conversation_summary:
                        fallback_context += f"Conversation Summary: {conversation_summary}\n\n"
                    if selected_blog_title:
                        fallback_context += f"User is asking about the blog post: '{selected_blog_title}'.\n\n"

                    if fallback_context:
                        messages = [
                            SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(context=fallback_context, question=user_query)),
                            HumanMessage(content=user_query)
                        ]
                    else:
                        messages = [
                            SystemMessage(content="You are a helpful and friendly AI assistant."),
                            HumanMessage(content=user_query)
                        ]
                    response = await model.ainvoke(messages)
                    response_text = response.content
                else:
                    context = "\n\n".join([
                        f"From '{r['blog_title']}':\n{r['content']}"
                        for r in results
                    ])
                    logger.debug(f"Retrieved {len(results)} context chunks for contextual QA")

                    messages = [
                        SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(context=context, question=user_query)),
                        HumanMessage(content=user_query)
                    ]
                    response = await model.ainvoke(messages)
                    response_text = response.content

                    for r in results:
                        unique_sources[r['blog_url']] = r['blog_title']
        else: 
            search_query_for_vector_store = user_query
            blog_url_filter = None
            if query_type == "direct" and state.get("selected_blog_title"):
                search_query_for_vector_store = state["selected_blog_title"]
                blog_url_filter = state.get("selected_blog_url")
                logger.info(f"Using selected blog title '{search_query_for_vector_store}' and URL '{blog_url_filter}' for vector search.")
            
            if not isinstance(search_query_for_vector_store, str):
                search_query_for_vector_store = str(search_query_for_vector_store)

            results = vector_store.search_blog_content(query=search_query_for_vector_store,
                                                       company_name=company_name,
                                                       blog_url=blog_url_filter,
                                                       limit=100) 

            if not results:
                logger.warning("No context found in vector store for query or selected blog.")
                response_text = "I couldn't find relevant blog content for that query. Please try a different topic or company."
            else:
                context = "\n\n".join([
                    f"From '{r['blog_title']}':\n{r['content']}"
                    for r in results
                ])

                logger.debug(f"Retrieved {len(results)} context chunks")

                messages = [
                    SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(context=context, question=user_query)),
                    HumanMessage(content=user_query)
                ]

                response = await model.ainvoke(messages)
                response_text = response.content

                for r in results:
                    unique_sources[r['blog_url']] = r['blog_title']

        logger.info("Response generated successfully")
        new_state = state.copy()
        new_state.update({
            "messages": state["messages"] + [AIMessage(content=response_text)],
            "step_count": state["step_count"] + 1,
        })

        if unique_sources:
            source_links = [f"[{title}]({url})" for url, title in unique_sources.items()]
            new_state["messages"][-1].content += f"\n\n*Sources: {', '.join(source_links)}*"

        return new_state
    except Exception as e:
        logger.error(f"Error in generate_response_node: {e}", exc_info=True)
        new_state = state.copy()
        new_state.update({
            "messages": state["messages"] + [AIMessage(content=f"An error occurred while generating the response: {e}")],
            "step_count": state["step_count"] + 1,
        })
        return new_state
