# Real-World Evaluation Framework: RSS Q&A Chatbot

Evaluating complex LLM pipelines—especially those utilizing autonomous routing (LangGraph) and Agentic Tool Calling—requires moving beyond simple unit tests. In production systems at companies like OpenAI or Meta, Evaluation ("Evals") is treated as a core engineering feature.

This blueprint outlines how to implement a **Deep, Agentic Evaluation Framework** for this project, transforming it into an enterprise-ready architecture.

---

## 1. Prerequisites: Decoupling & Observability

Before you can continuously evaluate an AI, you must decouple its execution from live environments to ensure tests are deterministic. 

### A. Telemetry & Tracing Integration
You cannot improve what you cannot see. The first step is replacing basic `logging.info` with structured telemetry.
- **Action**: Integrate an observability platform like **LangSmith** or **Arize Phoenix**. 
- **Why**: Since we use LangGraph, LangSmith allows you to replay the exact graph execution trace. You will be able to click on a user's question, see the exact embedding vector searched, and see the exact raw tool payload the LLM generated.

### B. Dependency Injection (Mocks)
Running Evals against live URLs and a live Qdrant cloud instance makes tests flaky and slow.
- **Action**: Refactor the LangGraph State (`AgentState`) or the global instances to accept generic interfaces. 
- **Implementation**: Instead of hardcoding `vector_store.search()`, inject `vector_store` into the `RunnableConfig` during compilation. During an Eval run, inject a `MockVectorStore` containing deterministic chunks so the exact same context is retrieved every single time. 

---

## 2. The Agentic "LLM-as-a-Judge" Evaluator

"Agentic Evaluation" means we don't use strict regex or keyword matching. We use a separate, highly capable model (e.g., GPT-4o or Claude 3.5 Sonnet) specifically prompted to act as an impartial judge.

### Recommended Tooling
We will use an enterprise evaluation framework such as **DeepEval** (by Confident AI), **Ragas**, or **LangChain Evaluation SDK**. For this blueprint, we assume **DeepEval**, as it integrates nicely with Pytest.

### The Evaluator Workflow
1. A python test script `test_agent.py` loads 100 historical User Queries (e.g., "What is Amazon Nova?").
2. The script triggers our RSS Q&A Chatbot offline.
3. The Chatbot returns the final `AIMessage` AND the retrieved `context` chunks.
4. The **Evaluator Agent** analyzes the Request, Context, and Output, calculating scores from 0.0 to 1.0 against distinct metrics.

---

## 3. The 4 Big Real-World Metrics

A production eval suite for this project must measure these four independent pillars:

### I. Routing Agent Accuracy (The Classifier)
Our custom `Classifier Node` decides the user's intent. If it routes a specific query to `discovery` instead of `direct search`, the entire flow fails.
- **Eval Metric**: Confusion Matrix & Accuracy.
- **Test Setup**: Feed 50 diverse queries. Ensure the LLM outputs `query_type: direct` dynamically when the expected dataset labels it as `direct`.

### II. Retrieval & Context Evaluation (RAG Quality)
Did the local HuggingFace Embeddings retrieve the right Jina Markdown chunks from Qdrant?
- **Context Precision**: Out of the 5 chunks retrieved, how many were actually useful? (Prevents polluting the LLM window).
- **Context Recall**: Can the Evaluator Agent find the answer in the retrieved context? If the answer is "Amazon Nova is 10T parameters" but the retrieved chunk doesn't state the parameter count, Recall is 0.

### III. Autonomous Tool-Calling Integrity (Deep Search)
The most advanced part of this bot is the autonomous `fetch_url_context` tool. We must test if the agent is utilizing its autonomy correctly.
- **Eval Metric (Tool Precision)**: Feed the bot a query requiring a deep-link read. 
- **Pass Condition**: The trace MUST show the LLM actively halting, executing the exact correct Tool string (`URL: https://...`), and processing the ToolMessage.
- **Fail Condition**: The LLM hallucinates an answer instead of using the tool, or it calls the tool with a fake/broken URL syntax.

### IV. Generation Verification (Faithfulness & Relevance)
The final response given to the user must be verified by the Judge Agent.
- **Faithfulness (Zero Hallucination)**: The Evaluator Agent reads the bot's response and compares it to the Qdrant Context. If the bot says *"AWS published this on Tuesday"*, but the context doesn't mention Tuesday, the Evaluator flags it as "Hallucinated: 0.0".
- **Relevance**: Did it answer the user's question, or just summarize the blog pointlessly?

---

## 4. Architecting the Eval Pipeline (The "Big" Implementation)

To build this, you would create an `evals/` directory parallel to `src/`:

```python
# evals/test_agent_faithfulness.py
import pytest
from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from src.agent import graph, AgentState

@pytest.mark.asyncio
async def test_blog_response_is_faithful():
    query = "How did OpenAI train GPT-4 based on the latest blog?"
    
    # 1. Run the actual agent
    final_state = await graph.ainvoke({"messages": [{"role": "user", "content": query}]})
    actual_response = final_state["messages"][-1].content
    
    # 2. Extract the context it actually retrieved (Requires state modification to output context)
    retrieved_chunks = [doc.page_content for doc in final_state["retrieved_docs"]]

    # 3. Formulate the LLM-as-a-Judge Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_response,
        retrieval_context=retrieved_chunks
    )

    # 4. Agentic Judge assesses hallucination using GPT-4
    metric = FaithfulnessMetric(threshold=0.8, model="gpt-4o")
    
    # 5. Assert Pass/Fail
    assert_test(test_case, [metric])
```

## 5. Next Steps for Optimization
If you want to implement this:
1. Wrap your FastApi `/api/chat` endpoints with LangSmith tracing.
2. Store every single user query in a lightweight SQLite "Ground Truth" database.
3. Once a week, an automated CI/CD GitHub Action should pull those 100 organic questions, run them through an Agentic Evaluator like DeepEval, and alert you in Slack if your Router Accuracy or Faithfulness dips below a 95% threshold!
