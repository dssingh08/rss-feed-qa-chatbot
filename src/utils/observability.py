""" Observability Setup for Arize Phoenix """

import logging
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def setup_arize_phoenix():
    """ Initialize Arize Phoenix for observability """
    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # We no longer launch the local ephemeral app here. We assume `start_phoenix_server.py` is running.
        # Register the local OpenTelemetry exporter, pointing to standard OTLP Phoenix receiver.
        tracer_provider = register(
            project_name="rss-qa-chatbot",
            endpoint="http://127.0.0.1:4317" # Phoenix gRPC transport
        )
        
        # Instrument LangChain to capture traces
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Successfully instrumented LangChain with Arize Phoenix (Standalone mode).")

        return None

    except ImportError:
        logger.warning(
            "Could not import phoenix or openinference modules. "
            "Please ensure they are installed to enable observability."
        )
    except Exception as e:
        logger.error(f"Failed to set up Arize Phoenix observability: {e}", exc_info=True)
