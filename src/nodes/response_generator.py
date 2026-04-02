""" Response Generator Node """
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.state import AgentState
from langchain_core.runnables import RunnableConfig
from src.models import get_response_model
from src.tools.vector_operations import vector_store
from src.tools.deep_scraper import fetch_url_context
from langchain_core.messages import ToolMessage
from src.utils.prompts import RESPONSE_GENERATION_PROMPT, GENERAL_RESPONSE_PROMPT
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


async def _invoke_with_tools(model, messages, config, max_iterations=3):
    """ Helper to run the tool calling loop for the model """
    model_with_tools = model.bind_tools([fetch_url_context])
    
    for _ in range(max_iterations):
        response = await model_with_tools.ainvoke(messages, config)
        
        if not response.tool_calls:
            return response
            
        # The model wants to use a tool, so we append its request to the history
        messages.append(response)
        
        for tool_call in response.tool_calls:
            if tool_call["name"] == "fetch_url_context":
                logger.info(f"LLM decided to fetch deep context for: {tool_call['args'].get('url')}")
                try:
                    tool_result = await fetch_url_context.ainvoke(tool_call)
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    tool_result = f"Error executing tool: {e}"
                
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
    
    # If we hit max iterations, just force a final response without tools
    return await model.ainvoke(messages, config)

async def generate_response_node(state: AgentState, config: RunnableConfig) -> AgentState:
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
    retrieved_docs = []

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
                blog_info = f"Current Blog: {selected_blog_title} ({selected_blog_url})\n" if selected_blog_title else ""
                messages = [
                    SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(blog_info=blog_info, context=context_for_general_query, question=user_query)),
                    HumanMessage(content=user_query)
                ]
            else:
                logger.debug("No specific context for general query, using basic prompt.")
                messages = [
                    SystemMessage(content=GENERAL_RESPONSE_PROMPT.format(question=user_query)),
                    HumanMessage(content=user_query)
                ]
            
            logger.debug(f"Messages for general query: {messages}")
            response = await _invoke_with_tools(model, messages, config)
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
                    SystemMessage(content=GENERAL_RESPONSE_PROMPT.format(question=user_query)),
                    HumanMessage(content=user_query)
                ]
                response = await _invoke_with_tools(model, messages, config)
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
                        blog_info = f"Current Blog: {selected_blog_title}\n"
                        messages = [
                            SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(blog_info=blog_info, context=fallback_context, question=user_query)),
                            HumanMessage(content=user_query)
                        ]
                    else:
                        messages = [
                            SystemMessage(content=GENERAL_RESPONSE_PROMPT.format(question=user_query)),
                            HumanMessage(content=user_query)
                        ]
                    response = await _invoke_with_tools(model, messages, config)
                    response_text = response.content
                else:
                    retrieved_docs.extend(results)
                    context = "\n\n".join([
                        f"From '{r['blog_title']}':\n{r['content']}"
                        for r in results
                    ])
                    logger.debug(f"Retrieved {len(results)} context chunks for contextual QA")

                    blog_info = f"Current Blog: {state.get('selected_blog_title')}\n"
                    messages = [
                        SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(blog_info=blog_info, context=context, question=user_query)),
                        HumanMessage(content=user_query)
                    ]
                    response = await _invoke_with_tools(model, messages, config)
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
                retrieved_docs.extend(results)
                context = "\n\n".join([
                    f"From '{r['blog_title']}':\n{r['content']}"
                    for r in results
                ])

                logger.debug(f"Retrieved {len(results)} context chunks")

                blog_info = f"Current Blog: {state.get('selected_blog_title')}\n"
                
                # Help the LLM understand that positional phrases (e.g. "5th blog") refer to the context
                llm_question = user_query
                if "blog" in user_query.lower() or any(char.isdigit() for char in user_query):
                    llm_question += f"\n[System Note: The user is referring to the current blog context ('{state.get('selected_blog_title')}'). Please fulfill their request using the text provided.]"
                
                messages = [
                    SystemMessage(content=RESPONSE_GENERATION_PROMPT.format(blog_info=blog_info, context=context, question=llm_question)),
                    HumanMessage(content=user_query)
                ]

                response = await _invoke_with_tools(model, messages, config)
                response_text = response.content

                for r in results:
                    unique_sources[r['blog_url']] = r['blog_title']

        logger.info("Response generated successfully")
        new_state = state.copy()
        new_state.update({
            "messages": state["messages"] + [AIMessage(content=response_text)],
            "step_count": state["step_count"] + 1,
            "retrieved_docs": retrieved_docs
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
