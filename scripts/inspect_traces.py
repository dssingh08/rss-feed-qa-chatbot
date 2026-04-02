import os
import json
import pandas as pd
import phoenix as px
from datetime import datetime

TRACE_IDS = [
    "06716c04028d687afdc23f78716f3beb",
    "51615e5f57a2c796329874292b581802"
]

def inspect_specific_traces(project_name="rss-qa-chatbot"):
    print(f"--- Inspecting Specific Traces: {TRACE_IDS} ---")
    
    try:
        client = px.Client(endpoint="http://127.0.0.1:6006")
        df = client.get_spans_dataframe(project_name=project_name)
    except Exception as e:
        print(f"Error connecting to Phoenix: {e}")
        return

    if df.empty:
        print("No traces found.")
        return

    report_path = f"debug_reports/trace_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs("debug_reports", exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Trace Inspection Report - {datetime.now().isoformat()}\n\n")
        
        for tid in TRACE_IDS:
            f.write(f"## Trace ID: `{tid}`\n")
            
            # Fetch all spans for this trace
            trace_spans = df[df['context.trace_id'] == tid]
            if trace_spans.empty:
                f.write("> [!WARNING]\n> Trace ID not found in current Phoenix database.\n\n")
                continue

            # Identify the Root (CHAIN) span
            root = trace_spans[trace_spans['span_kind'] == 'CHAIN']
            if root.empty:
                # Fallback to any span if CHAIN is missing (though it shouldn't be)
                root_span = trace_spans.iloc[0]
                f.write(f"**Root Span Kind:** {root_span.get('span_kind')} (CHAIN not found)\n")
            else:
                root_span = root.iloc[0]
                f.write(f"**Root Span Kind:** CHAIN\n")

            # Apply Extraction Logic
            input_val = root_span.get('attributes.input.value', 'N/A')
            output_val = root_span.get('attributes.output.value', 'N/A')
            
            # 1. Clean output
            output_str = str(output_val).strip()
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
                except: pass
            
            # Truncate like production
            output_str = output_str[:2000]

            # 2. Extract Context (Ultra-Strict Logic)
            retrieval_context = []
            seen_contents = set()
            total_chars = 0
            MAX_TOTAL_CONTEXT = 6000
            MAX_CHUNK_SIZE = 1500

            for _, sib in trace_spans.iterrows():
                if total_chars >= MAX_TOTAL_CONTEXT: break
                
                kind = str(sib.get('span_kind', '')).upper()
                potential_contents = []

                if kind == 'LLM':
                    messages = sib.get('attributes.llm.input_messages', [])
                    if isinstance(messages, list):
                        for msg in messages:
                            content = str(msg.get('message.content', ''))
                            if "Context from blog:" in content:
                                try:
                                    parts = content.split("Context from blog:")
                                    if len(parts) > 1:
                                        ctx_part = parts[1].split("User Question:")[0]
                                        if "Instructions:" in ctx_part:
                                            ctx_part = ctx_part.split("Instructions:")[0]
                                        val = ctx_part.strip()
                                        if val: potential_contents.append(val)
                                except: potential_contents.append(content)
                            
                            if msg.get('message.role') == 'tool' or 'tool_call' in str(msg.get('message.type', '')):
                                if content and content != "None": potential_contents.append(content)

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
                        else: potential_contents.append(str(sib_output))

                for c in potential_contents:
                    c_clean = c[:MAX_CHUNK_SIZE].strip()
                    if not c_clean: continue
                    content_hash = hash(c_clean[:100])
                    if content_hash not in seen_contents:
                        retrieval_context.append(c_clean)
                        seen_contents.add(content_hash)
                        total_chars += len(c_clean)
                        if total_chars >= MAX_TOTAL_CONTEXT: break

            f.write(f"### Extracted Data for Judge\n")
            f.write(f"**Input (User Question):**\n```text\n{input_val}\n```\n\n")
            f.write(f"**Actual Output (Sanitized & Truncated):**\n```text\n{output_str}\n```\n\n")
            f.write(f"**Retrieval Context Chunks:** {len(retrieval_context)} (Total chars: {total_chars})\n\n")
            
            for i, ctx in enumerate(retrieval_context):
                f.write(f"#### Context Chunk {i+1} ({len(ctx)} chars)\n")
                f.write("```text\n" + ctx + "\n```\n")

            f.write("### Raw Spans Inventory\n")
            f.write("| Name | Kind | Status |\n")
            f.write("| :--- | :--- | :--- |\n")
            for _, s in trace_spans.iterrows():
                f.write(f"| {s['name']} | {s.get('span_kind')} | {s.get('status_code')} |\n")
            
            f.write("\n---\n\n")

    print(f"Inspection report generated: {report_path}")
    return report_path

if __name__ == "__main__":
    inspect_specific_traces()
