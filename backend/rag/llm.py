import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI

_REQUIRED_ENV_VARS = (
    "DIAL_API_KEY",
    "DIAL_ENDPOINT",
    "DIAL_API_VERSION",
    "DIAL_DEPLOYMENT",
)


def get_llm() -> BaseChatModel:
    """Initialise AzureChatOpenAI from environment variables."""
    missing = [k for k in _REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise ValueError(
            f"Missing required environment variable(s): {', '.join(missing)}"
        )
    return AzureChatOpenAI(
        api_key=os.getenv("DIAL_API_KEY"),
        azure_endpoint=os.getenv("DIAL_ENDPOINT"),
        api_version=os.getenv("DIAL_API_VERSION"),
        azure_deployment=os.getenv("DIAL_DEPLOYMENT"),
    )
