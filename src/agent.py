import os
import time
import logging
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from crewai import Agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer

from src.config import MODEL_SERVICE, API_KEY, ANNUAL_REPORTS_DIR, setup_logging

# Set up logging
setup_logging()
class StockRecommendationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY, streaming=True, max_tokens=1500)
        self.tokenizer = self._initialize_tokenizer()
        self.db = self._initialize_database()
        self.summarize_chain = self._create_summarize_chain()
        self.recommend_chain = self._create_recommend_chain()

    def _initialize_tokenizer(self):
        logging.info("Initializing facebook/opt-350m tokenizer...")
        return AutoTokenizer.from_pretrained("facebook/opt-350m")

    def _initialize_database(self):
        logging.info("Loading PDF files and initializing database...")
        documents, self.potential_stocks = self._load_pdf_files(ANNUAL_REPORTS_DIR)
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(texts, embeddings)
        logging.info(f"Database initialized with {len(texts)} text chunks from {len(self.potential_stocks)} stocks")
        return db

    def _load_pdf_files(self, directory):
        documents = []
        stock_names = []
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'):
                stock_name = filename.split('.')[0]
                stock_names.append(stock_name)
                file_path = os.path.join(directory, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        return documents, stock_names

    def _create_summarize_chain(self):
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize the key financial information for the given stock based on the annual report data."),
            ("human", "Stock: {stock}\n\nAnnual Report Info: {annual_report_info}"),
            ("human", "Provide a brief summary of the key financial metrics and growth prospects.")
        ])
        return summarize_prompt | self.llm | StrOutputParser()

    def _create_recommend_chain(self):
        recommend_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial advisor. Recommend the top 3 stocks from the given summaries."),
            ("human", "Stock Summaries: {stock_summaries}"),
            ("human", "Based on these summaries, provide your top 3 stock recommendations for {query}. For each recommended stock, provide a brief justification. Avoid repeating information and keep each recommendation concise.")
        ])
        return recommend_prompt | self.llm | StrOutputParser()

    def num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def search_annual_reports(self, query: str, max_tokens: int = 1000) -> str:
        results = self.db.similarity_search(query, k=5)
        combined_text = ""
        for doc in results:
            if self.num_tokens(combined_text + doc.page_content) > max_tokens:
                break
            combined_text += doc.page_content + "\n\n"
        return combined_text.strip()

    def get_stock_summaries(self, stocks: List[str]) -> Dict[str, str]:
        summaries = {}
        for stock in stocks:
            logging.info(f"Summarizing {stock}...")
            annual_report_info = self.search_annual_reports(f"{stock} financial performance")
            summary = self.summarize_chain.invoke({"stock": stock, "annual_report_info": annual_report_info})
            summaries[stock] = summary
            logging.info(f"Summary for {stock}:\n{summary}\n")
        return summaries

    def get_stock_recommendations(self, query: str) -> str:
        logging.info("Starting stock recommendation process...")
        start_time = time.time()
        
        all_summaries = self.get_stock_summaries(self.potential_stocks)
        
        logging.info("Generating final recommendations...")
        result = self.recommend_chain.invoke({"stock_summaries": all_summaries, "query": query})
        
        end_time = time.time()
        logging.info(f"Recommendation process completed in {end_time - start_time:.2f} seconds")
        
        return result

# Simple executor function
def run_agent(query: str) -> str:
    agent = StockRecommendationAgent()
    return agent.get_stock_recommendations(query)

# Create a crewai-compatible agent
def create_crewai_agent():
    stock_agent = StockRecommendationAgent()
    return Agent(
        role="Stock Recommendation Specialist",
        goal="Provide accurate stock recommendations based on annual report analysis",
        backstory="An AI agent specialized in analyzing financial reports and recommending stocks.",
        tools=[stock_agent.get_stock_recommendations],
        verbose=True
    )

# Explicitly export the run_agent function
get_stock_recommendations = run_agent

# Example usage
if __name__ == "__main__":
    query = input("Enter your stock recommendation query: ")
    result = run_agent(query)
    print(result)