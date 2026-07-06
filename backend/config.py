import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
E2B_API_KEY = os.getenv("E2B_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# LangSmith tracing — set LANGCHAIN_TRACING_V2=true in .env to enable
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT", "autoanalyst-ai"))

# Check for placeholder API keys
PLACEHOLDER_KEYS = [
    "your_openai_api_key_here",
    "your_anthropic_api_key_here",
    "your_tavily_api_key_here",
    "your_e2b_api_key_here",
    "your_langsmith_api_key_here"
]

if OPENAI_API_KEY in PLACEHOLDER_KEYS or not OPENAI_API_KEY:
    logger.warning("⚠️  OPENAI_API_KEY is not set or is a placeholder. LLM calls will fail.")
if TAVILY_API_KEY in PLACEHOLDER_KEYS or not TAVILY_API_KEY:
    logger.warning("⚠️  TAVILY_API_KEY is not set or is a placeholder. Research will fail.")
if E2B_API_KEY in PLACEHOLDER_KEYS or not E2B_API_KEY:
    logger.warning("⚠️  E2B_API_KEY is not set or is a placeholder. Code execution will fail.")


def get_llm(temperature: float = 0):
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=LLM_MODEL, temperature=temperature, api_key=ANTHROPIC_API_KEY)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=LLM_MODEL, temperature=temperature, api_key=OPENAI_API_KEY)
