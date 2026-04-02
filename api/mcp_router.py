"""
MCP FastApi Endpoints
Exposes the tools and resources for use as an MCP connector.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Any
from src.tools.evaluation_tool import run_agent_eval
from src.tools.scraper_tool import scrape_blog

mcp_router = APIRouter(prefix="/mcp", tags=["MCP Connector"])

# Note: The official MCP standard requires Server Sentinel Events (SSE) or stdio transports. 
# This provides a basic HTTP abstraction for MCP-like tool interoperability.

class ToolDetails(BaseModel):
    name: str
    description: str

class MCPToolsResponse(BaseModel):
    tools: List[ToolDetails]

@mcp_router.get("/tools", response_model=MCPToolsResponse)
async def list_tools():
    """ List available MCP tools """
    return MCPToolsResponse(tools=[
        ToolDetails(name="run_agent_eval", description="Evaluate an agent's response using DeepEval metrics."),
        ToolDetails(name="scrape_blog", description="Fetch website content from a URL.")
    ])

class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]

@mcp_router.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """ Invoke an MCP tool """
    if request.name == "run_agent_eval":
        user_input = request.arguments.get("user_input", "")
        output = request.arguments.get("output", "")
        retrieved_context_json = request.arguments.get("retrieved_context_json", "[]")
        
        result = await run_agent_eval.ainvoke({
            "user_input": user_input, 
            "output": output, 
            "retrieved_context_json": retrieved_context_json
        })
        return {"content": result}
        
    elif request.name == "scrape_blog":
        url = request.arguments.get("url", "")
        result = await scrape_blog.ainvoke({"url": url})
        return {"content": result}

    return {"error": f"Tool '{request.name}' not found or unsupported via HTTP."}

@mcp_router.get("/resources")
async def list_resources():
    """ List available MCP resources (e.g., Evaluation traces and datasets) """
    return {"resources": [
        {"uri": "evals://datasets/smoke_test", "name": "Smoke Test Dataset", "type": "application/json"}
    ]}
