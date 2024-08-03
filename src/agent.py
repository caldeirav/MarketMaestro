from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

from src.config import MODEL_SERVICE, API_KEY, ANNUAL_REPORTS_DIR

# Initialize the LLM
llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY, streaming=True)

# Initialize RAG database
loader = DirectoryLoader(ANNUAL_REPORTS_DIR, glob="**/*.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)

# Define tools
def search_annual_reports(query: str) -> str:
    results = db.similarity_search(query, k=2)
    return "\n".join(doc.page_content for doc in results)

tools = [
    Tool(
        name="Annual Report Search",
        func=search_annual_reports,
        description="Useful for when you need to find information from company annual reports."
    )
]

# Define agent prompt
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a financial advisor specializing in stock recommendations. 
    Use the Annual Report Search tool to gather information about companies before making recommendations. 
    Always base your recommendations on the most recent financial data and market trends."""),
    ("human", "{input}"),
    ("human", "Thought: {agent_scratchpad}")
])

# Create the agent
agent = create_react_agent(llm, tools, agent_prompt)

# Create an agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Function to get stock recommendations
def get_stock_recommendations(query: str) -> dict:
    return agent_executor.invoke({"input": query})