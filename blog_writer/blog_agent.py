import os
import yaml
from typing import Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from blog_writer.tools.search_web import search_web

class BlogWritingAgent:
    """CrewAI agent that writes short blogs based on topics."""
    
    def __init__(self):
        # Initialize the language model
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Load configurations from YAML files
        self.agents_config = self._load_yaml('blog_writer/config/agents.yaml')
        self.tasks_config = self._load_yaml('blog_writer/config/tasks.yaml')
        
        # Create agents from config
        self.researcher = Agent(
            config=self.agents_config['researcher'],
            tools=[search_web()],
            llm=self.llm
        )
        
        self.writer = Agent(
            config=self.agents_config['writer'],
            llm=self.llm
        )
    
    def _load_yaml(self, filepath: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
    def write_blog(self, topic: str, word_count: int = 300) -> str:
        """Write a short blog post on the given topic."""
        try:
            # Create research task from config
            research_task = Task(
                config=self.tasks_config['research'],
                agent=self.researcher
            )
            
            # Create writing task from config
            writing_task = Task(
                config=self.tasks_config['write_blog'],
                agent=self.writer,
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
            result = crew.kickoff( inputs={"topic": topic, "word_count": word_count})
            
            return result
            
        except Exception as e:
            return f"Error writing blog: {str(e)}"

# Global instance
blog_agent = BlogWritingAgent() 