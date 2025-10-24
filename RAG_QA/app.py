import streamlit as st
import os
import time
from dotenv import load_dotenv

# --- LangChain & related imports ---
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai.embeddings import OpenAIEmbeddings

# --- Load environment variables ---
load_dotenv()

# --- Load API keys ---
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key or not openai_api_key:
    st.error("⚠️ Missing API keys! Please set GROQ_API_KEY and OPENAI_API_KEY in your .env file.")
    st.stop()

# --- Initialize LLM (Groq) ---
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# --- Define prompt template ---
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer the user's question as accurately as possible.

<context>
{context}
</context>

Question: {input}
""")

# --- Function to create FAISS vector embeddings ---
def create_vector_embeddings():
    """Loads PDFs, splits text, creates embeddings, and stores them in FAISS."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()  # Uses OpenAI API key
        st.session_state.loader = PyPDFDirectoryLoader("Policy")  # Folder containing PDFs
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50]  # Process first 50 docs for performance
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )
        st.success("✅ Vector database created successfully!")

# --- Streamlit UI ---
st.set_page_config(page_title="Groq-Powered PDF Chatbot", page_icon="📄")
st.title("📄 Groq-Powered PDF Q&A Chatbot")

st.markdown("""
Ask questions about your uploaded PDF documents.
Click **"Create Document Embeddings"** to index your files before asking questions.
""")

# --- Input field for the user's question ---
user_query = st.text_input("💬 Enter your question:")

# --- Button to create embeddings ---
if st.button("⚙️ Create Document Embeddings"):
    create_vector_embeddings()

# --- Process user query ---
if user_query:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create document embeddings first.")
    else:
        with st.spinner("🤔 Thinking..."):
            retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 3})

            # ✅ Modern replacement for StuffDocumentsChain
            doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)

            # ✅ Create retrieval + reasoning chain
            retrieval_chain = create_retrieval_chain(retriever, doc_chain)

            start = time.process_time()
            response = retrieval_chain.invoke({"input": user_query})
            elapsed = time.process_time() - start

            # --- Display results ---
            st.success(f"✅ Response generated in {elapsed:.2f} seconds!")
            answer = response.get("answer") or response.get("output_text", "")
            st.markdown(f"### 🧠 Answer:\n{answer}")

            # --- Show retrieved context documents ---
            with st.expander("🔍 Retrieved Documents"):
                context_docs = response.get("context", [])
                if not context_docs:
                    st.info("No documents were retrieved for this query.")
                else:
                    for i, doc in enumerate(context_docs, start=1):
                        source = doc.metadata.get("source", "Unknown")
                        st.markdown(f"**Document {i}:** {source}")
                        st.write(doc.page_content)
                        st.markdown("---")
