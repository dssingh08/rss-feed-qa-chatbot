""" LLM Model Configurations """

from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ChatOpenRouter(ChatOpenAI):
    """ Custom ChatOpenAI class for OpenRouter integration """

    def __init__(self, model_name: str, **kwargs):
        logger.info(f"Initializing OpenRouter model: {model_name}")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=settings.openrouter_api_key,
            model_name=model_name,
            **kwargs
        )


def get_internal_model(temperature: float = 0.0):
    """ Get Gemini 2.5 Flash for internal agent communication """
    logger.info("Creating internal Gemini 2.5 flash model ")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.google_api_key,
        temperature=temperature,
        convert_system_message_to_human=True
    )


def get_response_model(model_choice: str = "gemini", temperature: float = 0.7):
    """ Get model for user-facing response based on user selection """
    logger.info(f"Creating response model: {model_choice}")

    if model_choice == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gpt4o":
        return ChatOpenRouter(
            model_name="openai/gpt-4o-mini",
            temperature=temperature
        )
    elif model_choice == "llama":
        return ChatOpenRouter(
            model_name="meta-llama/llama-3.1-8b-instruct:free",
            temperature=temperature
        )
    else:
        logger.warning(f"Unknown model choice: {model_choice}, defaulting to Gemini")
        return get_response_model("gemini", temperature)
