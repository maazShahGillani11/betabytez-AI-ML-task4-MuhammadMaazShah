import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(page_title="CI/CD Knowledge Base", page_icon="🤖", layout="wide")
st.title("🤖 CI/CD Pipeline RAG Assistant")

api_key = os.getenv("GOOGLE_API_KEY")

if api_key:
    @st.cache_resource(show_spinner="Processing documentation...")
    def initialize_rag_system(key):
        txt_path = os.path.join("docs", "ci_cd_pipeline.txt")
        pdf_path = os.path.join("docs", "ci_cd_pipeline.pdf")
        vectorstore_dir = "vectorstore"

        # Local embeddings model setup
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # 1. Agar pehle se vectorstore local disk par saved hai, to wahan se load karein
        if os.path.exists(vectorstore_dir) and os.listdir(vectorstore_dir):
            vectorstore = FAISS.load_local(
                vectorstore_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            return vectorstore, len(vectorstore.docstore._dict)

        # 2. Agar saved nahi hai, to document parh kar new index banayein
        if os.path.exists(txt_path):
            loader = TextLoader(txt_path, encoding="utf-8")
        elif os.path.exists(pdf_path):
            loader = PyPDFLoader(pdf_path)
        else:
            st.error("❌ No document found in 'docs/' folder!")
            return None, 0

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)

        # Create FAISS Vector Store
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Local disk (vectorstore/ folder) par permanently save karein
        vectorstore.save_local(vectorstore_dir)
        
        return vectorstore, len(chunks)

    vectorstore, total_chunks = initialize_rag_system(api_key)

    if vectorstore:
        st.sidebar.success(f"✅ Document Indexed into {total_chunks} Chunks!")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key, 
            temperature=0.0
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        system_prompt = (
            "You are a strict technical Q&A assistant specializing in the provided CI/CD document.\n"
            "Use ONLY the retrieved context below to answer the question.\n"
            "If the answer is NOT in the context, respond strictly: "
            "'I'm sorry, but the provided document does not contain information to answer this question.'\n\n"
            "Retrieved Context:\n{context}\n\n"
            "Question: {question}"
        )

        prompt = ChatPromptTemplate.from_template(system_prompt)
        format_docs = lambda docs: "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask a question about CI/CD pipelines..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    sources = retriever.invoke(user_query)
                    answer = rag_chain.invoke(user_query)

                    st.markdown(answer)
                    
                    if sources and "does not contain information" not in answer:
                        with st.expander("📌 Source Citations"):
                            for idx, source in enumerate(sources):
                                st.caption(f"**Chunk {idx+1}:** {source.page_content}")

            st.session_state.messages.append({"role": "assistant", "content": answer})