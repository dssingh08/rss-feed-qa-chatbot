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


def get_internal_model(model_choice: str = "gemini", temperature: float = 0.0):
    """ Get internal agent model based on user selection """
    logger.info(f"Creating internal model: {model_choice}")
    
    if model_choice in ["gemini", "gemini-2.5", "gemini-2.5-flash"]:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-2.5-flash-lite":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-3-flash":
        return ChatGoogleGenerativeAI(
            model="models/gemini-3.0-flash", # API typically uses `gemini-3-flash` or similar, fallback handled locally
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-1.5-flash":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
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
            model_name="meta-llama/llama-4-maverick:free",
            temperature=temperature
        )
    else:
        logger.warning(f"Unknown model choice: {model_choice}, defaulting to Gemini")
        return get_internal_model("gemini", temperature)


def get_response_model(model_choice: str = "gemini", temperature: float = 0.7):
    """ Get model for user-facing response based on user selection """
    logger.info(f"Creating response model: {model_choice}")

    if model_choice in ["gemini", "gemini-2.5", "gemini-2.5-flash"]:
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-2.5-flash-lite":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-3-flash":
        return ChatGoogleGenerativeAI(
            model="models/gemini-3.0-flash",
            google_api_key=settings.google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True
        )
    elif model_choice == "gemini-1.5-flash":
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
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
            model_name="meta-llama/llama-4-maverick:free",
            temperature=temperature
        )
    else:
        logger.warning(f"Unknown model choice: {model_choice}, defaulting to Gemini")
        return get_response_model("gemini", temperature)


def get_eval_judge(temperature: float = 0.0):
    """ Get the LLM used for DeepEval as a judge """
    logger.info("Creating OpenRouter eval judge model (StepFun)")
    # Using the user-requested StepFun model on OpenRouter
    return ChatOpenRouter(
        model_name="qwen/qwen3.6-plus-preview:free",
        temperature=temperature
    )
