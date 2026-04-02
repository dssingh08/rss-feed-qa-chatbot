""" FastAPI Server """

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict
import uuid
import os
import sys
import platform
import asyncio

if platform.system() == "Windows":
    # Required for Playwright to run subprocesses on Windows in asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add project root to sys.path so modules like 'src' can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agent import graph
from src.state import AgentState
from src.utils.logger import setup_logger
from src.utils.observability import setup_arize_phoenix
from langchain_core.messages import HumanMessage
from api.evaluation_router import eval_router
from api.mcp_router import mcp_router

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Lifespan context manager """
    logger.info("Starting RSS Q&A Chatbot API")
    session = setup_arize_phoenix()
    yield
    logger.info("Shutting down API")
    if session:
        # Avoid hanging if running as part of uvicorn
        pass


app = FastAPI(
    title="RSS Q&A Chatbot API",
    description="Intelligent RSS blog Q&A system with LangGraph",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(eval_router)
app.include_router(mcp_router)

active_connections: Dict[str, WebSocket] = {}


@app.get("/")
async def root():
    """ Health check """
    return {"status": "healthy", "service": "RSS Q&A Chatbot"}


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "active_connections": len(active_connections)
    }

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """ WebSocket endpoint for chat """
    active_connections[user_id] = websocket

    logger.info(f"WebSocket connected: user_id={user_id}")

    config = {"configurable": {"thread_id": user_id}}

    try:
        await websocket.send_json({
            "type": "system",
            "content": "Connected to RSS Q&A Chatbot. Ask me about blogs from companies like Google, OpenAI, Amazon and more!"
        })

        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")
            model_choice = data.get("model", "gemini")

            logger.info(f"Received message from {user_id}: {user_message[:50]}...")

            initial_state = {
                "messages": [HumanMessage(content=user_message)],
                "user_id": user_id,
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
                "response_model": model_choice,
                "step_count": 0
            }

            async for event in graph.astream(initial_state, config=config):
                logger.debug(f"Agent even: {event.keys()}")

                for node_name, node_ouput in event.items():
                    if "messages" in node_ouput and node_ouput["messages"]:
                        last_message = node_ouput["messages"][-1]

                        await websocket.send_json({
                            "type": "message",
                            "content": last_message.content,
                            "node": node_name
                        })
            
            logger.info(f"Completed processing for {user_id}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {user_id}")
        del active_connections[user_id]
    
    except Exception as e:
        logger.error(f"WebSocket error for {user_id}: {str(e)}")
        try:
            await websocket.close()
        except:
            pass
        finally:
            if user_id in active_connections:
                del active_connections[user_id]


class ChatRequest(BaseModel):
    user_id: str
    message: str
    model: str = "gemini"

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """ REST API endpoint for chat to replace direct frontend graph invocation """
    from langchain_core.messages import HumanMessage
    
    logger.info(f"Received API chat request from {request.user_id}")
    config = {"configurable": {"thread_id": request.user_id}}
    
    # Check if thread exists in checkpointer
    current_state = graph.get_state(config)
    if not current_state.values:
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id,
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
            "response_model": request.model,
            "step_count": 0
        }
    else:
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "response_model": request.model
        }
    
    try:
        final_state = await graph.ainvoke(input_state, config=config)
        
        ai_messages = [msg for msg in final_state["messages"] if msg.type == "ai"]
        if ai_messages:
            content = ai_messages[-1].content
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        text_parts.append(block["text"])
                    elif isinstance(block, str):
                        text_parts.append(block)
                response_text = "".join(text_parts)
            else:
                response_text = str(content)
        else:
            response_text = "I couldn't process that request. Please try again."
            
        return ChatResponse(response=response_text)
    except Exception as e:
        logger.error(f"Error processing chat API request: {str(e)}", exc_info=True)
        return ChatResponse(response="An error occurred while processing your request.")

@app.post("/api/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    """ REST API endpoint for streaming chat"""
    from langchain_core.messages import HumanMessage
    import json
    
    logger.info(f"Received API streaming chat request from {request.user_id}")
    config = {"configurable": {"thread_id": request.user_id}}
    
    current_state = graph.get_state(config)
    if not current_state.values:
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_id": request.user_id,
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
            "response_model": request.model,
            "step_count": 0
        }
    else:
        input_state = {
            "messages": [HumanMessage(content=request.message)],
            "response_model": request.model
        }
    
    async def event_generator():
        has_streamed = False
        try:
            async for event in graph.astream_events(input_state, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    if event.get("metadata", {}).get("langgraph_node") == "Generate Response":
                        chunk = event["data"]["chunk"].content
                        if chunk:
                            # Handle formats where content is a list of blocks (like Gemini 3)
                            if isinstance(chunk, list):
                                text_parts = []
                                for block in chunk:
                                    if isinstance(block, dict) and "text" in block:
                                        text_parts.append(block["text"])
                                    elif isinstance(block, str):
                                        text_parts.append(block)
                                processed_chunk = "".join(text_parts)
                            else:
                                processed_chunk = str(chunk)
                                
                            if processed_chunk:
                                has_streamed = True
                                yield f"data: {json.dumps({'chunk': processed_chunk})}\n\n"
                elif event["event"] == "on_chain_end" and event["name"] == "Generate Response":
                    if not has_streamed:
                        output = event["data"].get("output", {})
                        if isinstance(output, dict) and "messages" in output and output["messages"]:
                            content = output["messages"][-1].content
                            import asyncio
                            for word in content.split(" "):
                                yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
                                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Error in stream generator: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
