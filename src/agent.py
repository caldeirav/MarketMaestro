import os
import time
import logging
import re
from typing import List, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer
from crewai import Agent

from src.config import MODEL_SERVICE, API_KEY, ANNUAL_REPORTS_DIR, setup_logging

# Set up logging
setup_logging()

class StockRecommendationAgent(Agent):
    def __init__(self):
        llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY, streaming=True, max_tokens=500)
        super().__init__(
            name="Stock Recommendation Agent",
            role="Financial Analyst",
            goal="Provide accurate stock recommendations based on financial data and market trends.",
            backstory="I am an AI agent specialized in analyzing financial data and providing stock recommendations.",
            llm=llm,
            verbose=True
        )
        self._initialize()

    def _initialize(self):
        self._tokenizer = self._initialize_tokenizer()
        self._db = self._initialize_database()
        self._sys_prompt = "You are an AI language model. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        self._summarize_chain = self._create_summarize_chain()
        self._recommend_chain = self._create_recommend_chain()

    def _initialize_tokenizer(self):
        logging.info("Initializing facebook/opt-350m tokenizer...")
        return AutoTokenizer.from_pretrained("facebook/opt-350m")

    def _initialize_database(self):
        logging.info("Loading PDF files and initializing database...")
        documents = []
        for filename in os.listdir(ANNUAL_REPORTS_DIR):
            if filename.endswith('.pdf'):
                stock = self._extract_stock_name(filename)
                file_path = os.path.join(ANNUAL_REPORTS_DIR, filename)
                loader = PyPDFLoader(file_path)
                for doc in loader.load():
                    doc.metadata['source'] = filename
                    doc.metadata['stock'] = stock
                documents.extend(loader.load())

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma.from_documents(texts, embeddings)
        logging.info(f"Database initialized with {len(texts)} text chunks")
        return db

    def _create_summarize_chain(self):
        summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", self._sys_prompt),
            ("human", "Summarize the key financial information for the given stock based on the annual report data, focusing on aspects relevant to the query. Provide a concise summary (250 words or less) of the key financial metrics, growth prospects, and any initiative or investment directly related to the query."),
            ("human", "Stock: {stock}\n\nAnnual Report Info: {annual_report_info}\n\nQuery: {query}")
        ])
        return summarize_prompt | self.llm | StrOutputParser()

    def _create_recommend_chain(self):
        recommend_prompt = ChatPromptTemplate.from_messages([
            ("system", self._sys_prompt),
            ("human", "You are a financial advisor. Based on the following concise stock summaries and the user's query, recommend one stock that best fits the criteria. Provide a brief justification for your recommendation."),
            ("human", "Stock Summaries: {stock_summaries}"),
            ("human", "User Query: {query}"),
            ("human", "Recommend one stock and explain why it's the best fit for the query in 150 words or less.")
        ])
        return recommend_prompt | self.llm | StrOutputParser()

    def _extract_stock_name(self, filename: str) -> str:
        match = re.match(r'([a-zA-Z]+)-\d+\.pdf', filename)
        if match:
            return match.group(1).upper()
        return ''

    def num_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text))

    def search_annual_reports(self, query: str, max_tokens: int = 1000) -> str:
        results = self._db.similarity_search(query, k=5)
        combined_text = ""
        for doc in results:
            if self.num_tokens(combined_text + doc.page_content) > max_tokens:
                break
            combined_text += doc.page_content + "\n\n"
        return combined_text.strip()

    def get_stock_summaries(self, query: str) -> Dict[str, str]:
        logging.info(f"Generating concise summaries for all stocks based on query: {query}")
        
        all_stocks = [self._extract_stock_name(filename) for filename in os.listdir(ANNUAL_REPORTS_DIR) if filename.endswith('.pdf')]
        logging.info(f"Total stocks found: {len(all_stocks)}")
        
        summaries = {}
        for stock in all_stocks:
            logging.info(f"Summarizing {stock}...")
            annual_report_info = self.search_annual_reports(f"{stock} financial performance related to {query}")
            summary = self._summarize_chain.invoke({
                "stock": stock, 
                "annual_report_info": annual_report_info,
                "query": query
            })
            summaries[stock] = summary
            logging.info(f"Concise summary for {stock}:\n{summary}\n")
        return summaries

    def get_stock_recommendations(self, query: str) -> str:
        logging.info("Starting stock recommendation process...")
        start_time = time.time()
        
        formatted_query = f'<|system|>\n{self._sys_prompt}\n<|user|>\n{query}\n<|assistant|>\n'
        
        summaries = self.get_stock_summaries(query)
        
        logging.info("Generating recommendations based on query...")
        result = self._recommend_chain.invoke({
            "stock_summaries": summaries,
            "query": formatted_query
        })
        
        result = result.replace('<|endoftext|>', '').strip()
        
        end_time = time.time()
        logging.info(f"Recommendation process completed in {end_time - start_time:.2f} seconds")
        
        return result

    def execute_task(self, task, context=None):
        query = task if isinstance(task, str) else task.description
        return self.get_stock_recommendations(query)

# Example usage
if __name__ == "__main__":
    agent = StockRecommendationAgent()
    query = input("Enter your stock recommendation query: ")
    result = agent.execute_task(query)
    print(result)