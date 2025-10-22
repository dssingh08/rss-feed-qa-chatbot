""" FastAPI Server """

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict
import uuid
from src.agent import graph
from src.state import AgentState
from src.utils.logger import setup_logger
from langchain_core.messages import HumanMessage

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Lifespan context manager """
    logger.info("Starting RSS Q&A Chatbot API")
    yield
    logger.info("Shutting down API")


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


if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )