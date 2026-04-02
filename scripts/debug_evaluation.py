import os
import json
import pandas as pd
import phoenix as px
from datetime import datetime

def debug_evaluation_traces(project_name="rss-qa-chatbot", limit=10):
    print(f"--- Debugging Evaluation Traces for Project: {project_name} ---")
    
    try:
        client = px.Client(endpoint="http://127.0.0.1:6006")
        df = client.get_spans_dataframe(project_name=project_name)
    except Exception as e:
        print(f"Error connecting to Phoenix: {e}")
        return

    if df.empty:
        print("No traces found.")
        return

    # 1. Identify DeepEval Spans
    eval_spans = df[df['name'].str.contains("deep_eval_", na=False)]
    
    if eval_spans.empty:
        print("No 'deep_eval_*' spans found. Have you run an evaluation yet?")
        return

    parent_col = next((c for c in df.columns if c.endswith('parent_id')), 'parent_id')
    span_id_col = next((c for c in df.columns if c.endswith('span_id')), 'span_id')

    report_path = f"debug_reports/eval_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs("debug_reports", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# DeepEval Debugging Report - {datetime.now().isoformat()}\n\n")
        f.write(f"Total Evaluation Spans Found: {len(eval_spans)}\n\n")

        for _, eval_span in eval_spans.iterrows():
            span_id = eval_span.get(span_id_col)
            f.write(f"## Evaluation Span: {eval_span['name']} ({span_id})\n")
            f.write(f"**Status:** {eval_span.get('status_code', 'Unknown')}\n")
            
            trace_id = eval_span.get('context.trace_id')
            
            # Enhanced Context Recovery (Matching Router Logic)
            retrieval_context = []
            if trace_id:
                siblings = df[df['context.trace_id'] == trace_id]
                for _, sib in siblings.iterrows():
                    kind = str(sib.get('span_kind', '')).upper()
                    
                    # Look for content in LLM prompts or Tool Outputs
                    if kind == 'LLM':
                        messages = sib.get('attributes.llm.input_messages', [])
                        if isinstance(messages, list):
                            for msg in messages:
                                content = str(msg.get('message.content', ''))
                                if "Context from blog:" in content:
                                    try:
                                        parts = content.split("Context from blog:")
                                        if len(parts) > 1:
                                            # Simple heuristic to extract the context block
                                            ctx_part = parts[1].split("User Question:")[0]
                                            if ctx_part.strip() and ctx_part.strip() not in retrieval_context:
                                                retrieval_context.append(ctx_part.strip())
                                    except: pass
                                
                                if msg.get('message.role') == 'tool' or 'tool_call' in str(msg.get('message.type', '')):
                                    if content and content not in retrieval_context:
                                        retrieval_context.append(content)

                    elif kind == 'TOOL':
                        out = sib.get('attributes.output.value', '')
                        if out and out != "None":
                            retrieval_context.append(str(out))

            f.write(f"**Extracted Context Chunks:** {len(retrieval_context)}\n\n")

            # Extract Attributes
            f.write("### Recorded Attributes\n")
            f.write("```json\n")
            eval_attrs = {k: v for k, v in eval_span.items() if k.startswith('attributes.eval.')}
            f.write(json.dumps(eval_attrs, indent=2) + "\n")
            f.write("```\n\n")

            # Find judge's LLM calls
            children = df[df[parent_col] == span_id]
            f.write(f"### Judge LLM Calls ({len(children)})\n")
            
            for _, child in children.iterrows():
                f.write(f"#### Child Span: {child['name']}\n")
                f.write("**Input (Prompt to Judge):**\n")
                f.write("```text\n")
                input_val = next((child.get(c) for c in df.columns if 'input.value' in c or 'llm.prompts' in c), 'N/A')
                f.write(str(input_val) + "\n")
                f.write("```\n")
                f.write("**Output (Judge Verdict):**\n")
                f.write("```text\n")
                output_val = next((child.get(c) for c in df.columns if 'output.value' in c or 'llm.output_messages' in c), 'N/A')
                f.write(str(output_val) + "\n")
                f.write("```\n\n")

            f.write("---\n\n")

    print(f"Debug report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    debug_evaluation_traces()
