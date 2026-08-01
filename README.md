# CI/CD Pipeline RAG Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system built with **LangChain**, **FAISS**, **Streamlit**, and **Google Gemini (gemini-2.5-flash)**. This assistant delivers strict, context-grounded Q&A over CI/CD Pipeline technical documentation with zero hallucination.
# Technical Architecture & Workflow
[ CI/CD Document ] ➡️ [ PyPDF / Text Loader ] ➡️ [ Recursive Chunking ]
                                                           │
                                                           ▼
[ Streamlit UI ] ⬅️ [ Gemini 2.5 LLM ] ⬅️ [ FAISS Index (MiniLM) ]
1. Document Loading: Ingests raw text or PDF documents defining CI/CD methodologies, stages, tools, and deployment strategies.
2. Text Chunking: Uses RecursiveCharacterTextSplitter (chunk size: 800, overlap: 150) to split documents into semantically coherent contexts.
3. Vector Indexing: Generates dense vector representations using all-MiniLM-L6-v2 and indexes them inside a fast in-memory FAISS vector store.
4. Context Retrieval: Fetches Top-K ($k=3$) relevant document chunks per user query via similarity search.
5. Strict Prompt Engineering: Enforces non-hallucination boundaries via system prompt grounding.
# Tech Stack

 Framework**: LangChain (LCEL)
 Vector Database: FAISS (Facebook AI Similarity Search)
 Embeddings Model: HuggingFace (`all-MiniLM-L6-v2`)
 LLM: Google Gemini (`gemini-2.5-flash`)
 Frontend / UI: Streamlit
 Environment Handling: `python-dotenv`
# Setup & Execution Guide
# 1. Repository Setup & Dependencies
  Clone the repository and install the verified dependency requirements:
  ```bash
git clone <your-repo-url>
cd week-4-5
pip install -r requirements.txt
2. Environment Configuration
Create a .env file in the root directory:

Code snippet
GOOGLE_API_KEY=your_gemini_api_key_here
3. Launching Application
Run the Streamlit app with clean watcher config:

Bash
streamlit run app.py --server.fileWatcherType none
Test Results & Proof of Grounding
In-Scope Queries (Ground Truth Verification)
Query: What is the difference between Continuous Integration, Continuous Delivery, and Continuous Deployment?
Result: Provides precise definitions, highlights manual approval gates for Delivery vs. automatic production rollout for Deployment.
Query: Explain Canary deployment and its benefits.
Result: Explains incremental traffic shifting (e.g., 10% rollout) to minimize production release risk.
Out-of-Scope Queries (Zero Hallucination Verification)
Query: What is the recipe for making a pepperoni pizza?
Result: "I'm sorry, but the provided document does not contain information to answer this question."
Query: What is Quantum Computing and how does it work?
Result: "I'm sorry, but the provided document does not contain information to answer this question."

Folder Structure
WEEK 4-5/
├── .env
├── app.py
├── README.md
├── requirements.txt
├── docs/
│   ├── ci_cd_pipeline.txt
│   └── ci_cd_pipeline.pdf
└── vectorstore/