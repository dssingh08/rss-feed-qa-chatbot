"""
Streamlit Frontend
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import asyncio
import platform
import logging
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logging.getLogger().setLevel(logging.CRITICAL)

import streamlit as st
import json
import traceback
from websockets import connect
from src.config import settings
from src.utils.logger import setup_logger
from src.agent import graph
from src.state import AgentState 
from langchain_core.runnables import RunnableConfig 

logger = setup_logger(__name__)

logging.getLogger("streamlit").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


st.set_page_config(
    page_title="RSS Q&A Chatbot",
    layout="wide"
)

st.title("RSS Q&A Chatbot")
st.markdown("Ask me about blogs from companies like Google, OpenAI, Amazon, Microsoft and more!")

with st.sidebar:
    st.header("Settings")
    
    model_choice = st.selectbox(
        "Response Model",
        options=["gemini", "gpt4o", "llama"],
        format_func=lambda x: {
            "gemini": "Gemini 2.5 Flash",
            "gpt4o": "GPT-4o",
            "llama": "Llama 3.1"
        }[x]
    )
    
    st.divider()
    
    st.header("Available Companies")
    companies = ["Google", "OpenAI", "Amazon", "Microsoft", "Meta", "Anthropic", "Langchain"]
    for company in companies:
        st.markdown(f"- {company}")
    
    st.divider()
    
    st.header("Example Queries")
    st.markdown("""
    **Type 1 (Discovery):**
    - "Is there a new blog by Google?"
    - "Show me latest OpenAI posts"
    
    **Type 2 (Direct):**
    - "Explain LLM hallucination by OpenAI"
    - "Tell me about Google's Gemini model"
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_id" not in st.session_state:
    import uuid
    st.session_state.user_id = str(uuid.uuid4())

    if "agent_state" not in st.session_state:
        st.session_state.agent_state = {
            "messages": [],
            "user_id": st.session_state.user_id,
            "summary": "",
            "query_type": "unknown",
            "company_name": None,
            "topic": None,
            "blog_titles": None,
            "selected_blog_url": None,
            "selected_blog_title": None,
            "vector_store_ids": [],
            "should_scrape": False,
            "scraper_reason": None,
            "response_model": "gemini",
            "step_count": 0
        }


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about company blogs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            from langchain_core.messages import HumanMessage
            
            config = {"configurable": {"thread_id": st.session_state.user_id}}
            
            st.session_state.agent_state["messages"].append(HumanMessage(content=prompt))
            st.session_state.agent_state["response_model"] = model_choice

            initial_agent_state = AgentState(**st.session_state.agent_state)

            runnable_config: RunnableConfig = {"configurable": {"thread_id": st.session_state.user_id}}

            final_state = asyncio.run(graph.ainvoke(initial_agent_state, config=runnable_config))
            
            st.session_state.agent_state.update(final_state)

            ai_messages = [msg for msg in final_state["messages"] if msg.type == "ai"]
            if ai_messages:
                response = ai_messages[-1].content
            else:
                response = "I couldn't process that request. Please try again."
            
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            full_traceback = traceback.format_exc()
            logger.error(f"Error processing message: {str(e)}\n{full_traceback}")
            
            error_msg = f"An unexpected error occurred. Please try again. Details: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
