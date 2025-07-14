import os
from typing import Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from crewai.tools import BaseTool
import re

class search_web(BaseTool):
    """Search the web for information about a topic."""
    name: str = "search_web"
    description: str = "Search the web for information about a topic"
    def _run(self, query: str) -> str:
        search_tool = DuckDuckGoSearchRun()
        result = search_tool.run(query)
        return result

class BlogWritingAgent:
    """CrewAI agent that writes short blogs based on topics."""
    
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create the research agent
        self.researcher = Agent(
            role='Research Analyst',
            goal='Conduct thorough research on the given topic to gather accurate and relevant information',
            backstory="""You are an expert research analyst with years of experience in gathering 
            and analyzing information from various sources. You have a keen eye for identifying 
            credible sources and extracting key insights.""",
            verbose=True,
            allow_delegation=False,
            tools=[search_web()],
            llm=self.llm
        )
        
        # Create the writer agent
        self.writer = Agent(
            role='Content Writer',
            goal='Write engaging, informative, and well-structured short blog posts based on research',
            backstory="""You are a skilled content writer with expertise in creating engaging blog posts. 
            You have a talent for making complex topics accessible and interesting to readers. 
            You always write in a clear, concise, and engaging style.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def write_blog(self, topic: str, word_count: int = 300) -> str:
        """Write a short blog post on the given topic."""
        try:
            # Create research task
            research_task = Task(
                description=f"""Research the topic: {topic}
                
                Your research should cover:
                1. Key facts and statistics about the topic
                2. Current trends or developments
                3. Different perspectives or viewpoints
                4. Practical applications or examples
                
                Use the search_web function to find recent, relevant, and credible information.
                Provide a comprehensive summary of your findings.""",
                agent=self.researcher,
                expected_output="A detailed research summary with key findings, facts, and insights about the topic."
            )
            
            # Create writing task
            writing_task = Task(
                description=f"""Write a short, engaging blog post about: {topic}
                
                Requirements:
                - Word count: approximately {word_count} words
                - Engaging introduction that hooks the reader
                - Clear, well-structured content with subheadings
                - Practical insights or actionable takeaways
                - Professional yet conversational tone
                - Include relevant examples or case studies
                - End with a compelling conclusion
                
                Use the research provided to create accurate, informative content.
                Make the content accessible to a general audience while maintaining depth.""",
                agent=self.writer,
                expected_output=f"A well-written blog post of approximately {word_count} words with proper structure and engaging content.",
                context=[research_task]
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.researcher, self.writer],
                tasks=[research_task, writing_task],
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            return result
            
        except Exception as e:
            return f"Error writing blog: {str(e)}"


# Global instance
blog_agent = BlogWritingAgent() 