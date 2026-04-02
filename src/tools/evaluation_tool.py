""" Evaluation Tool and Metrics Engine """

from langchain_core.tools import tool
try:
    from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase
    from deepeval.models import DeepEvalBaseLLM
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from src.models import get_eval_judge
from src.utils.logger import setup_logger
import json

logger = setup_logger(__name__)

if DEEPEVAL_AVAILABLE:
    class DeepEvalOpenRouterJudge(DeepEvalBaseLLM):
        def __init__(self, model):
            self.model = model

        def load_model(self):
            return self.model

        def generate(self, prompt: str) -> str:
            return self.model.invoke(prompt).content

        async def a_generate(self, prompt: str) -> str:
            import asyncio
            max_retries = 3
            last_err = None
            for attempt in range(max_retries):
                try:
                    res = await self.model.ainvoke(prompt)
                    return res.content
                except Exception as e:
                    last_err = e
                    logger.warning(f"OpenRouter retry {attempt+1}/{max_retries} due to: {e}")
                    await asyncio.sleep(5)  # Backoff
            
            logger.error(f"OpenRouter failed after {max_retries} attempts: {last_err}")
            raise last_err

        def get_model_name(self):
            return "OpenRouter Judge"


async def evaluate_interaction(user_input: str, output: str, retrieved_context: list[str]) -> dict:
    """ Evaluates a single interaction using DeepEval Metrics """
    if not DEEPEVAL_AVAILABLE:
        return {"error": "DeepEval is not installed or importing failed."}

    logger.info("Starting evaluation for interaction")
    judge_model = get_eval_judge()
    deepeval_judge = DeepEvalOpenRouterJudge(judge_model)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=output,
        retrieval_context=retrieved_context
    )

    faithfulness = FaithfulnessMetric(threshold=0.7, model=deepeval_judge, include_reason=True)
    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=deepeval_judge, include_reason=True)

    metrics = [faithfulness, answer_relevancy]
    results = {}
    
    # Safety: If context is empty, Faithfulness will fail or be meaningless.
    # We add a dummy context or skip it if it's a non-retrieval query.
    if not retrieved_context:
        logger.warning("Empty retrieval context provided for evaluation. Faithfulness metric may be unreliable.")
        # Optional: If it's a conversational query, we might want to skip Faithfulness
        # For now, we'll let it run but log the warning.

    from opentelemetry import trace
    tracer = trace.get_tracer(__name__)

    for metric in metrics:
        with tracer.start_as_current_span(f"deep_eval_{metric.__name__}") as span:
            try:
                logger.info(f"Measuring {metric.__name__}")
                await metric.a_measure(test_case)
                
                passed = metric.is_successful()
                score = metric.score
                reason = metric.reason
                
                # Attach the actual evaluation scores to the trace for Arize Phoenix!
                span.set_attribute("eval.name", metric.__name__)
                span.set_attribute("eval.score", float(score))
                span.set_attribute("eval.passed", passed)
                span.set_attribute("eval.reason", str(reason))
                
                results[metric.__name__] = {
                    "score": score,
                    "reason": reason,
                    "passed": passed
                }
            except Exception as e:
                span.record_exception(e)
                logger.error(f"Error evaluating {metric.__name__}: {e}", exc_info=True)
                results[metric.__name__] = {
                    "error": str(e)
                }

    return results

@tool
async def run_agent_eval(user_input: str, output: str, retrieved_context_json: str) -> str:
    """
    Evaluates the agent's response against retrieved context.
    Use this to self-correct or assess quality.
    Args:
        user_input: The original user question.
        output: The generated response.
        retrieved_context_json: Array of text chunks used as context encoded in JSON.
    """
    logger.info("Running Agent Evaluation Tool")
    try:
        retrieved_context = json.loads(retrieved_context_json)
        if not isinstance(retrieved_context, list):
            retrieved_context = [str(retrieved_context)]
    except json.JSONDecodeError:
        retrieved_context = [retrieved_context_json]

    results = await evaluate_interaction(user_input, output, retrieved_context)
    
    if "error" in results:
        return f"Evaluation failed: {results['error']}"

    report = "Evaluation Results:\n"
    for name, data in results.items():
        if "error" in data:
            report += f"- {name}: Error ({data['error']})\n"
        else:
            status = "PASS" if data['passed'] else "FAIL"
            report += f"- {name}: {status} (Score: {data['score']})\n  Reasoning: {data['reason']}\n"
    
    return report
