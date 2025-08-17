"""
Search Web Tool for CrewAI Agents

This tool allows agents to search the web for information using DuckDuckGo.
"""

from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

class search_web(BaseTool):
    """Search the web for information about a topic."""
    name: str = "search_web"
    description: str = "Search the web for information about a topic"
    
    def _run(self, query: str) -> str:
        """Execute the web search."""
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.run(query)
        return result 