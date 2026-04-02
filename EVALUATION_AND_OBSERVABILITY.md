# Framework Integration: Evaluation, Observability & MCP

This document summarizes the changes made to the RSS Q&A Chatbot project to implement enterprise-grade evaluation, observability tracking, and expose capabilities via a Model Context Protocol (MCP) compatible API.

---

## 1. Observability (Tracing & Telemetry)
The agent currently uses **Arize Phoenix** for local runtime tracing via OpenTelemetry instrumentation.

### What is tracked?
- **LangGraph Workflows**: Every node transition (Classifier -> Search -> Response Generator) is recorded.
- **LLM Calls**: Inputs, system prompts, output generation, and token usage for all LLM interactions are captured.
- **Tools**: Tool execution inputs and resulting outputs are tracked. Specifically, the agent's autonomous use of the `fetch_url_context` deep scraping tool.
- **Vector Retrieval**: RAG retrieval steps are instrumented to see exactly what context chunks were fetched from Qdrant.

**How to run and view traces persistently:**
We use a centralized standalone architecture. Open a NEW terminal and run:
```bash
python start_phoenix_server.py
```
This script will spin up a permanent Phoenix backend that saves its data to a local `.phoenix_data/` folder in your project root. Any app you build can now send OpenTelemetry metrics to `http://127.0.0.1:4317`, and you will never lose your history! 

You can view the main dashboard at exactly `http://localhost:6006`.

---

## 2. LLM Evaluator (DeepEval)
We utilize a custom Judge model powered by a free reasoning LLM via OpenRouter (`meta-llama/llama-3.1-8b-instruct:free`).

### Added Metrics
We've integrated **DeepEval** to automate quality checks on the agent:
1. **Faithfulness**: Measures if the generated output hallucinated information not present in the retrieved text chunks.
2. **Answer Relevancy**: Checks if the generated response actually addresses the user's initial query payload.

These evaluations can be run offline using the `pytest` suite in `tests/evals/test_evaluations.py`, or interactively through the new API endpoints.

---

## 3. New API Endpoints & MCP Connector

To allow the project to be integrated into another product (e.g., as an MCP Connector), new endpoints have been created.

### A. Comprehensive Evaluation Endpoint
Evaluates either a single generation trace or batches of previous chats using the DeepEval Judge.

- **Endpoint**: `POST /api/evaluate`
- **Logic**: 
    If `actual_output` is missing, the API will automatically invoke the LangGraph Agent to generate it, dynamically fetch the RAG context, and then score the output.
- **Payload Structure**:
```json
{
  "mode": "single", // Choices: "single" or "batch"
  "dataset_source": "provided", // Choices: "provided" (JSON in request) or "local" (server-side file)
  "dataset_name": "smoke_test", // Only required if dataset_source == "local"
  "cases": [ // Only required if dataset_source == "provided"
    {
      "input": "User's query string",
      "actual_output": "Optional: Pre-generated chatbot response",
      "retrieval_context": ["Optional: Associated RAG context chunk list"] 
    }
  ]
}
```

### B. Standardized MCP Tool Endpoint
Exposes specific LangChain agent capabilities so external agents can utilize them.

- **Endpoint**: `GET /mcp/tools` (Returns a list of available AI tools including self-evaluation and blog scraping).
- **Endpoint**: `POST /mcp/tools/call`
- **Payload Structure**:
```json
{
  "name": "run_agent_eval",  // Choices for tools: "run_agent_eval" or "scrape_blog"
  "arguments": {
    "user_input": "...",
    "output": "...",
    "retrieved_context_json": "[\"chunk1\", \"chunk2\"]"
  }
}
```

### C. Standardized MCP Resource Endpoint
Exposes static artifacts to external MCP systems.

- **Endpoint**: `GET /mcp/resources`
- Returns predefined resources like evaluation datasets available on the host machine.
