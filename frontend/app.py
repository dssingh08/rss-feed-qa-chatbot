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
# Removed direct graph dependencies

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
        options=["gemini", "gemini-2.5-flash-lite", "gemini-3-flash", "gemini-1.5-flash", "gpt4o", "llama"],
        format_func=lambda x: {
            "gemini": "Gemini 2.5 Flash",
            "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
            "gemini-3-flash": "Gemini 3 Flash",
            "gemini-1.5-flash": "Gemini 1.5 Flash",
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
        st.session_state.agent_state = {}


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
            payload = {
                "user_id": st.session_state.user_id,
                "message": prompt,
                "model": model_choice
            }
            
            import requests
            
            def stream_response():
                try:
                    with requests.post(
                        "http://127.0.0.1:8000/api/chat/stream",
                        json=payload,
                        stream=True,
                        timeout=120.0
                    ) as r:
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: "):
                                    data_str = line[6:]
                                    try:
                                        data_json = json.loads(data_str)
                                        if "chunk" in data_json:
                                            yield data_json["chunk"]
                                        elif "error" in data_json:
                                            yield f"\n\n**Error:** {data_json['error']}"
                                    except json.JSONDecodeError:
                                        pass
                except Exception as e:
                    yield f"An error occurred: {str(e)}"
            
            response = message_placeholder.write_stream(stream_response())
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            full_traceback = traceback.format_exc()
            logger.error(f"Error processing message: {str(e)}\n{full_traceback}")
            
            error_msg = f"An unexpected error occurred. Please try again. Details: {str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
