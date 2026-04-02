# RSS Q&A Chatbot: Technical Architecture & deep Dive

## 1. Project Overview
The **RSS Q&A Chatbot** is an intelligent, agent-driven application that allows users to discover, read, and cross-examine the latest technical engineering blogs from top companies (e.g., Google, AWS, OpenAI, Meta, Anthropic). 

Instead of just returning summaries, the bot utilizes a state-machine architecture to classify user intent, fetch live RSS feeds, semantically search vector databases, and perform autonomous "deep scraping" to formulate highly accurate, context-aware answers.

---

## 2. Technical Stack & Architecture

### Frontend
- **Streamlit**: Chosen for rapid UI prototyping, allowing native chat-like message rendering and session state management.

### Backend Infrastructure
- **FastAPI**: Serves the REST API and handles real-time **Server-Sent Events (SSE)** for streaming the LLM's response chunk-by-chunk to the Streamlit UI.
- **LangGraph**: The core cognitive engine. Instead of a linear prompt chain, LangGraph models the AI as a state machine with nodes (Classifier, Scraper, Responder, etc.) and conditional routing edges.
- **LLM Factory**: Built around Google GenAI (specifically optimized for `gemini-2.5-flash` variants). Includes a factory pattern to dynamically select high-RPM/TPM models from the UI to avoid rate limits.

### Data & Memory Layer
- **Qdrant (Cloud)**: The Vector Database. Chosen for its high performance and payload filtering capabilities.
- **Local HuggingFace Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) via Python to embed text chunks.

---

## 3. Core Features & Agentic Flow

When a user sends a message, it flows through this cognitive cycle:

1. **Classifier Node**: Analyzes the query using a system prompt to determine the user's intent. Routes to:
   - `discovery`: ("What's new at OpenAI?") -> Fetches RSS feeds.
   - `blog_selection`: ("Read the 2nd one") -> Marks the URL for extraction.
   - `direct`: ("Explain the DPO blog by AWS") -> Searches Qdrant for semantic matches.
   - `contextual_qa`: ("What dataset did they use in that?") -> Queries existing context.
2. **RSS Fetcher / Processor Nodes**: Parses standard XML/RSS feeds into clean lists of titles, URLs, and pub-dates.
3. **Scraper Node**: Transforms target URLs into LLM-readable text.
4. **Response Generator Node**: Retrieves vector chunks and formulates the final answer.

---

## 4. Key Optimizations & "Why" We Built It This Way

Throughout development, several architecture bottlenecks were encountered and solved through strategic engineering:

### A. Local CPU Embeddings vs. Cloud API
- **The Problem**: Initially, the app used Gemini's `models/embedding-001` API. However, splitting a 20,000-word tech blog into chunks and hitting a free-tier Cloud API quickly exhausted quotas, resulting in `404 Not Found` or `429 Too Many Requests` mapping errors that crashed the bot.
- **The Solution**: Completely removed Cloud API embeddings. Migrated to `all-MiniLM-L6-v2` running entirely offline on the local CPU via the `langchain-huggingface` library. 
- **Why**: CPU embeddings guarantee zero rate limits, zero API costs, and execute incredibly fast for 384-dimension vectors.

### B. Web Scraping: Jina Reader vs. Playwright
- **The Problem**: Heavy JavaScript frameworks (SPAs) and strict bot-protection (Cloudflare `403 Forbidden` on OpenAI/Anthropic blogs) blocked standard `BeautifulSoup` scraping. Attempting to deploy `Playwright` (Headless Chromium) introduced severe Windows `asyncio` event-loop crashes and made the backend bloated.
- **The Solution**: Integrated the **Jina Reader API (`https://r.jina.ai/`)**.
- **Why**: Jina acts as a high-quality proxy that natively executes JavaScript, bypasses bot captchas, and most importantly, strips away all HTML DOM noise (Navbars, Ads, Footers) returning pure, clean Markdown. This vastly improves the LLM's comprehension.

### C. Agentic Deep Context vs. Background Spidering
- **The Goal**: If a blog says, *"We used the [XYZ Architecture](url) to scale,"* we want the user to be able to ask *"What is the XYZ Architecture?"* even if it wasn't explained in the primary blog.
- **Abandoned Approach**: "Lazy Background Spidering" (automatically extracting all links and scraping them in the background). Rejected because it wastes memory, triggers anti-bot rate limits, and fills Qdrant with useless PR links.
- **The Solution**: **On-Demand Tool Calling**. We created a LangChain `@tool` (`fetch_url_context`) and bound it to the `Generate Response` node. 
- **Why**: If the user asks a deep question, the LLM reads the primary Markdown context, spots the hyperlinked URL, natively pauses, executes the Scraper Tool on that exact URL, reads the newly fetched Markdown, and resumes generating the final answer. It costs zero storage bloat unless explicitly required by the user's curiosity.

### D. Qdrant URL Caching & The "Depth Overlap" Upgrade
- **The Problem**: Re-scraping the same blog every time a user references it takes 2-4 seconds.
- **The Solution**: The Scraper Node checks `qdrant.check_blog_exists(url)` before firing HTTP requests. If found, it skips scraping entirely (0ms latency).
- **The Deep Upgrade Feature**: If the LLM dynamically scrapes a hyperlinked documentation page, it is stored in Qdrant as `depth=1` (Supplementary). If another user later asks to read that explicit documentation page as their main query, Qdrant recognizes the URL cache hit, skips the scrape, and our Response Node dynamically upgrades its label from "Supplementary" to "Primary Context" in the prompt.

### E. Frontend/Backend Decoupling (The SSE Migration)
- **The Problem**: Originally, the LangGraph `graph.stream()` was invoked directly inside Streamlit's UI thread. Streamlit reruns scripts from top-to-bottom on every user interaction, which frequently severed the graph generator and corrupted the conversational memory states.
- **The Solution**: Ripped LangGraph out of the frontend entirely. Wrapped it in a dedicated FastAPI server, communicating back to Streamlit via REST endpoints and Server-Sent Events (SSE) for type-writer token streaming.
- **Why**: Ensures the cognitive engine runs in a stable, isolated environment unaffected by frontend UI clicks.
