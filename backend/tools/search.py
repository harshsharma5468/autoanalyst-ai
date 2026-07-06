from langchain_community.tools.tavily_search import TavilySearchResults
from backend.config import TAVILY_API_KEY
import os

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY or ""

tavily_tool = TavilySearchResults(
    max_results=6,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
)
