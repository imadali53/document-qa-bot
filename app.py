
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile
import os

st.set_page_config(page_title="Document Q&A Bot", page_icon="📄")
st.title("📄 Document Q&A Bot")
st.caption("Upload a document and ask questions about it — powered by RAG")

# Sidebar for API key
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
st.sidebar.markdown("[Get free Groq API key](https://console.groq.com)")

# File uploader
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])

if uploaded_file and groq_api_key:
    with st.spinner("Processing document..."):
        # Save uploaded file temporarily
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load document
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Build LLM and chain
        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don\'t have that information in the document."

Context:
{context}

Question:
{question}

Answer:
""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        os.unlink(tmp_path)
        st.success(f"Document processed — {len(chunks)} chunks ready ✅")

    # Chat interface
    st.subheader("Ask a question about your document")
    question = st.text_input("Your question:")

    if question:
        with st.spinner("Searching document..."):
            answer = rag_chain.invoke(question)
        st.markdown(f"**Answer:** {answer}")

elif uploaded_file and not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
else:
    st.info("Upload a document to get started.")
