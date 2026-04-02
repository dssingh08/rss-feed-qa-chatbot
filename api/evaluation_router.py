""" API endpoints for Evaluation """

import os
import json
import logging
import uuid
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from src.tools.evaluation_tool import evaluate_interaction
from src.agent import graph
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

eval_router = APIRouter(prefix="/api/evaluate", tags=["Evaluation"])

class EvaluationCase(BaseModel):
    input: str
    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None

class EvaluationRequest(BaseModel):
    mode: Literal["single", "batch"] = Field(
        "single", description="Whether to evaluate a single output or a batch dataset."
    )
    dataset_source: Literal["provided", "local", "phoenix_traces"] = Field(
        "provided", description="Source of the evaluation data."
    )
    dataset_name: Optional[str] = Field(
        None, description="Name of the server-side dataset if local (e.g. 'smoke_test')"
    )
    cases: Optional[List[EvaluationCase]] = Field(
        None, description="The test cases to evaluate if source is 'provided'"
    )
    trace_count: Optional[int] = Field(
        None, description="Number of traces to fetch if dataset_source is 'phoenix_traces'"
    )
    trace_kind_filter: Optional[str] = Field(
        None, description="Optional Span Kind to filter by from Phoenix (e.g. 'LLM', 'CHAIN', 'TOOL')"
    )

class EvaluationResult(BaseModel):
    input: str
    actual_output: str
    scores: dict
    passed: bool

class BatchEvaluationResponse(BaseModel):
    total_cases: int
    passed_cases: int
    average_scores: dict
    details: List[EvaluationResult]


@eval_router.post("", response_model=BatchEvaluationResponse)
async def evaluate_agent(request: EvaluationRequest):
    """
    Run Agent Evaluation Metrics.
    Supports single or batch modes. Allows using local server predefined datasets or JSON provided arrays.
    """
    cases_to_run = []

    # 1. Load Data
    if request.dataset_source == "provided":
        if not request.cases:
            raise HTTPException(status_code=400, detail="Must provide 'cases' when source is 'provided'.")
        cases_to_run = request.cases
    elif request.dataset_source == "local":
        if not request.dataset_name:
            raise HTTPException(status_code=400, detail="Must provide 'dataset_name' when source is 'local'.")
        
        dataset_path = os.path.join(os.path.dirname(__file__), "..", "tests", "evals", "datasets", f"{request.dataset_name}.json")
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_cases = json.load(f)
                cases_to_run = [EvaluationCase(**c) for c in raw_cases]
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Dataset error: {str(e)}")
            

    elif request.dataset_source == "phoenix_traces":
        if not request.trace_count:
            raise HTTPException(status_code=400, detail="Must provide 'trace_count' when source is 'phoenix_traces'.")
        
        try:
            import phoenix as px
            import pandas as pd
            
            client = px.Client(endpoint="http://127.0.0.1:6006")
            # Pull span dataframe
            try:
                df = client.get_spans_dataframe(project_name="rss-qa-chatbot")
            except Exception:
                try: 
                    df = client.get_spans_dataframe(project_name="default")
                except Exception as e:
                    raise HTTPException(status_code=404, detail=f"No Phoenix traces accessible: {e}")

            if df.empty:
                raise HTTPException(status_code=404, detail="No traces found in Phoenix.")
                
            # Default to CHAIN if not specified, as it represents the full request
            kind_filter = (request.trace_kind_filter or "CHAIN").lower()
            
            # Filter for target spans
            if 'span_kind' in df.columns:
                target_df = df[df['span_kind'].str.lower() == kind_filter]
            else:
                target_df = df # Fallback
                
            # Sort by most recent
            if 'start_time' in target_df.columns:
                target_df = target_df.sort_values(by='start_time', ascending=False)
                
            # Limit safely
            n_traces = min(len(target_df), request.trace_count)
            target_df = target_df.head(n_traces)
            
            for _, row in target_df.iterrows():
                trace_id = row.get('context.trace_id')
                input_str = ""
                output_str = ""
                retrieval_context = []
                
                # 1. Extract basic I/O from the CHAIN span itself
                input_str = str(row.get('attributes.input.value', ''))
                output_str = str(row.get('attributes.output.value', ''))
                
                # 2. Extract Context from SIBLING/CHILD spans
                # In this graph, documents aren't via a separate Retriever, but formatted into prompts or fetched via tools.
                # 2. Extract Context from SIBLING/CHILD spans
                # In this graph, documents aren't via a separate Retriever, but formatted into prompts or fetched via tools.
                if trace_id:
                    siblings = df[df['context.trace_id'] == trace_id]
                    seen_contents = set()
                    total_chars = 0
                    MAX_TOTAL_CONTEXT = 6000  # Ultra-strict for StepFun
                    MAX_CHUNK_SIZE = 1500

                    for _, sib in siblings.iterrows():
                        if total_chars >= MAX_TOTAL_CONTEXT:
                            break

                        kind = str(sib.get('span_kind', '')).upper()
                        
                        potential_contents = []

                        # Case A: Content was formatted into an LLM System Message
                        if kind == 'LLM':
                            messages = sib.get('attributes.llm.input_messages', [])
                            if isinstance(messages, list):
                                for msg in messages:
                                    content = str(msg.get('message.content', ''))
                                    # Look for LangGraph-style formatted context in prompts
                                    if "Context from blog:" in content:
                                        try:
                                            parts = content.split("Context from blog:")
                                            if len(parts) > 1:
                                                # Refined split to avoid capturing instructions or the question
                                                ctx_part = parts[1].split("User Question:")[0]
                                                if "Instructions:" in ctx_part:
                                                    ctx_part = ctx_part.split("Instructions:")[0]
                                                
                                                val = ctx_part.strip()
                                                if val:
                                                    potential_contents.append(val)
                                        except:
                                            potential_contents.append(content)
                                    
                                    # Also look for ToolMessage content in history
                                    if msg.get('message.role') == 'tool' or 'tool_call' in str(msg.get('message.type', '')):
                                        if content and content != "None":
                                            potential_contents.append(content)

                        # Case B: Content in TOOL output (e.g. from fetch_url_context)
                        elif kind == 'TOOL':
                            sib_output = sib.get('attributes.output.value', '')
                            if sib_output and sib_output != "None":
                                if isinstance(sib_output, str) and sib_output.strip().startswith('['):
                                    try:
                                        docs = json.loads(sib_output)
                                        if isinstance(docs, list):
                                            for d in docs:
                                                val = d.get('page_content', d.get('content', d.get('text', str(d)))) if isinstance(d, dict) else str(d)
                                                potential_contents.append(str(val))
                                    except: pass
                                else:
                                    potential_contents.append(str(sib_output))

                        # Process and add unique contents
                        for c in potential_contents:
                            # Prune each chunk
                            c_clean = c[:MAX_CHUNK_SIZE].strip()
                            if not c_clean: continue
                            
                            # Deduplicate by hashing the first 100 chars to avoid very similar overlaps
                            content_hash = hash(c_clean[:100])
                            if content_hash not in seen_contents:
                                retrieval_context.append(c_clean)
                                seen_contents.add(content_hash)
                                total_chars += len(c_clean)
                                if total_chars >= MAX_TOTAL_CONTEXT:
                                    break
                
                # 3. Clean up noisy LLM metadata from output_str
                # If output_str looks like a LangChain/Phoenix JSON blob, extract the core content
                output_str = output_str.strip()
                if output_str.startswith('{'):
                    try:
                        data = json.loads(output_str)
                        if 'generations' in data:
                            gen = data['generations'][0][0]
                            output_str = gen.get('text', '')
                            if not output_str and 'message' in gen:
                                msg = gen.get('message', {})
                                output_str = msg.get('kwargs', {}).get('content', msg.get('content', output_str))
                        elif 'output' in data:
                            output_str = data['output']
                        elif 'text' in data:
                            output_str = data['text']
                        elif 'kwargs' in data:
                            output_str = data['kwargs'].get('content', output_str)
                    except: pass
                
                # Truncate output_str for stability (StepFun limit)
                output_str = str(output_str)[:2000]

                # Final fallback for empty but valid traces
                if not input_str or input_str == "None" or input_str == "{}":
                    input_str = f"Trace: {trace_id[:8] if trace_id else 'Unknown'}"
                    
                cases_to_run.append(EvaluationCase(
                    input=input_str,
                    actual_output=output_str if output_str and output_str != "None" else None,
                    retrieval_context=retrieval_context
                ))
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to extract from Phoenix: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to fetch traces from Phoenix: {str(e)}")


    if request.mode == "single" and len(cases_to_run) > 1:
        cases_to_run = [cases_to_run[0]]

    # 2. Run generation if necessary and execute evaluation
    results = []
    
    for case in cases_to_run:
        actual_output = case.actual_output
        retrieval_context = case.retrieval_context or []

        # If missing actual output, we run the agent to get it
        if not actual_output:
            logger.info(f"Generating output for input: {case.input}")
            session_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": session_id}}
            initial_state = {
                "messages": [HumanMessage(content=case.input)],
                "user_id": session_id,
                "query_type": "unknown",
                "step_count": 0
            }
            try:
                final_state = await graph.ainvoke(initial_state, config=config)
                # Parse generated response
                ai_messages = [m for m in final_state.get("messages", []) if m.type == "ai"]
                if ai_messages:
                    actual_output = ai_messages[-1].content
                else:
                    actual_output = "No response generated."
                
                # Extract context used
                retrieved_docs_raw = final_state.get("retrieved_docs", [])
                for doc in retrieved_docs_raw:
                    content = doc.get("content", "")
                    if content and content not in retrieval_context:
                        retrieval_context.append(content)

            except Exception as e:
                logger.error(f"Error during agent trace: {e}")
                actual_output = f"Error generating response: {e}"

        logger.info(f"Evaluating trace for input: {case.input}")
        eval_scores = await evaluate_interaction(case.input, actual_output, retrieval_context)

        # Determine overall pass
        passed_all = True
        for metric, data in eval_scores.items():
            if "error" in data or not data.get("passed", False):
                passed_all = False
                break
                
        results.append(EvaluationResult(
            input=case.input,
            actual_output=actual_output,
            scores=eval_scores,
            passed=passed_all
        ))

    # 3. Aggregate results
    aggregates = {}
    pass_count = 0
    total = len(results)

    for r in results:
        if r.passed:
            pass_count += 1
        for m, d in r.scores.items():
            if "score" in d:
                if m not in aggregates:
                    aggregates[m] = []
                aggregates[m].append(d["score"])

    average_scores = {k: sum(v)/len(v) for k, v in aggregates.items() if len(v) > 0}

    return BatchEvaluationResponse(
        total_cases=total,
        passed_cases=pass_count,
        average_scores=average_scores,
        details=results
    )
