import os
import time
import logging
from typing import List, Dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import tiktoken

from src.config import MODEL_SERVICE, API_KEY, ANNUAL_REPORTS_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the LLM
llm = ChatOpenAI(base_url=MODEL_SERVICE, api_key=API_KEY, streaming=True, max_tokens=1500)

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-2")

def num_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Function to load multiple PDF files and return a list of stock names
def load_pdf_files(directory):
    documents = []
    stock_names = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            stock_name = filename.split('.')[0]  # Assuming filename format is "STOCKNAME.pdf"
            stock_names.append(stock_name)
            file_path = os.path.join(directory, filename)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents, stock_names

# Initialize RAG database and get stock names
logging.info("Loading PDF files and initializing database...")
documents, potential_stocks = load_pdf_files(ANNUAL_REPORTS_DIR)
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use HuggingFaceEmbeddings with the latest sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)
logging.info(f"Database initialized with {len(texts)} text chunks from {len(potential_stocks)} stocks")

# Define search function
def search_annual_reports(query: str, max_tokens: int = 1000) -> str:
    results = db.similarity_search(query, k=5)
    combined_text = ""
    for doc in results:
        if num_tokens(combined_text + doc.page_content) > max_tokens:
            break
        combined_text += doc.page_content + "\n\n"
    return combined_text.strip()

# Define summarization prompt
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the key financial information for the given stock based on the annual report data."),
    ("human", "Stock: {stock}\n\nAnnual Report Info: {annual_report_info}"),
    ("human", "Provide a brief summary of the key financial metrics and growth prospects.")
])

# Create a summarization chain
summarize_chain = summarize_prompt | llm | StrOutputParser()

# Function to get stock summaries
def get_stock_summaries(stocks: List[str]) -> Dict[str, str]:
    summaries = {}
    for stock in stocks:
        logging.info(f"Summarizing {stock}...")
        annual_report_info = search_annual_reports(f"{stock} financial performance")
        summary = summarize_chain.invoke({"stock": stock, "annual_report_info": annual_report_info})
        summaries[stock] = summary
        logging.info(f"Summary for {stock}:\n{summary}\n")
    return summaries

# Define recommendation prompt
recommend_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a financial advisor. Recommend the top 3 stocks from the given summaries."),
    ("human", "Stock Summaries: {stock_summaries}"),
    ("human", "Based on these summaries, provide your top 3 stock recommendations for {query}. For each recommended stock, provide a brief justification. Avoid repeating information and keep each recommendation concise.")
])

# Create a recommendation chain
recommend_chain = recommend_prompt | llm | StrOutputParser()

# Function to get stock recommendations
def get_stock_recommendations(query: str) -> Dict[str, str]:
    logging.info("Starting stock recommendation process...")
    start_time = time.time()
    
    all_summaries = get_stock_summaries(potential_stocks)
    
    logging.info("Generating final recommendations...")
    result = recommend_chain.invoke({"stock_summaries": all_summaries, "query": query})
    
    end_time = time.time()
    logging.info(f"Recommendation process completed in {end_time - start_time:.2f} seconds")
    
    return {"output": result}

# Simple executor function
def run_agent(query: str) -> None:
    result = get_stock_recommendations(query)
    print(result['output'])