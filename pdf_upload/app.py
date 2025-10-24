import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document  # ✅ Needed for fixes

# LLM setup
from langchain_community.chat_models import ChatOpenAI  # fallback if Groq not available

# -------------------------------
# INITIAL SETUP
# -------------------------------
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

st.set_page_config(page_title="AI Policy Comparison Advisor", page_icon="📄")
st.title("📄 Policy Advisor Bot")
st.write("Upload **two policy PDFs**, customize what matters most, and get a data-driven recommendation.")

# -------------------------------
# API SETUP
# -------------------------------
api_key = st.text_input("Enter your Groq API key (or OpenAI key):", type="password")
if not api_key:
    st.warning("Please enter your API key to continue.")
    st.stop()

try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant")  # or llama-3-70b if available
except ImportError:
    llm = ChatOpenAI(openai_api_key=api_key, model_name="gpt-3.5-turbo")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device": "cpu"}
)

# -------------------------------
# FILE UPLOADS
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    policy_a = st.file_uploader("Upload Policy A", type="pdf", key="policy_a")
with col2:
    policy_b = st.file_uploader("Upload Policy B", type="pdf", key="policy_b")

if not (policy_a and policy_b):
    st.info("Upload both policies to start comparison.")
    st.stop()

# -------------------------------
# USER WEIGHTS
# -------------------------------
st.subheader("⚖️ Set Your Priorities")
col1, col2, col3 = st.columns(3)
coverage_weight = col1.slider("Coverage Importance", 0, 100, 30)
cost_weight = col2.slider("Cost Importance", 0, 100, 25)
exclusion_weight = col3.slider("Exclusions Importance", 0, 100, 20)
claims_weight = col1.slider("Claims Process Importance", 0, 100, 15)
benefits_weight = col2.slider("Customer Benefits Importance", 0, 100, 10)

total_weight = coverage_weight + cost_weight + exclusion_weight + claims_weight + benefits_weight
if total_weight == 0:
    st.warning("Please assign at least one non-zero weight.")
    st.stop()

weights = {
    "Coverage": coverage_weight / total_weight,
    "Cost": cost_weight / total_weight,
    "Exclusions": exclusion_weight / total_weight,
    "Claims": claims_weight / total_weight,
    "Benefits": benefits_weight / total_weight,
}

# -------------------------------
# EXPLANATION MODE
# -------------------------------
st.subheader("🗣️ Choose Explanation Mode")
explanation_mode = st.radio(
    "How should the assistant explain the results?",
    options=["Simple", "Professional"],
    index=0,
    horizontal=True
)

# -------------------------------
# DOCUMENT LOADING AND EMBEDDING
# -------------------------------
def load_and_embed(pdf_file):
    temp_path = f"./temp_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())

    docs = PyPDFLoader(temp_path).load()
    os.remove(temp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever_a = load_and_embed(policy_a)
retriever_b = load_and_embed(policy_b)

# -------------------------------
# EXTRACTION CHAIN
# -------------------------------
extract_prompt = ChatPromptTemplate.from_template("""
You are a policy analyst. Extract the following details from this policy text:

- Coverage details
- Premium / Cost information
- Major exclusions
- Claims process
- Additional or customer benefits

Summarize clearly and concisely.

Policy Text:
{context}
""")

extract_chain = create_stuff_documents_chain(llm, extract_prompt)

# -------------------------------
# COMPARISON PROMPT
# -------------------------------
tone = "in a simple and easy-to-understand way" if explanation_mode == "Simple" else "in a detailed and professional tone"

comparison_prompt = ChatPromptTemplate.from_template("""
You are an expert insurance policy evaluator.

Use a {tone} explanation style.

Compare **Policy A** and **Policy B** based on these weighted criteria:
- Coverage ({coverage_pct}%)
- Cost ({cost_pct}%)
- Exclusions ({exclusions_pct}%)
- Claims Process ({claims_pct}%)
- Benefits ({benefits_pct}%)

Below is the extracted information from both policies:
{context}

For each criterion:
1. Give a score (1–10) for Policy A and Policy B.
2. Justify briefly.
3. Compute weighted total scores.
4. Recommend which policy is overall better and why (in 3 bullet points).

Structure your response as:

### Comparison Summary Table
| Criteria | Policy A (Score) | Policy B (Score) | Notes |
|-----------|------------------|------------------|-------|
| ... | ... | ... | ... |

### Weighted Totals
- Policy A Total: X
- Policy B Total: Y

### Recommendation
...
""").partial(
    tone=tone,
    coverage_pct=f"{weights['Coverage']*100:.0f}",
    cost_pct=f"{weights['Cost']*100:.0f}",
    exclusions_pct=f"{weights['Exclusions']*100:.0f}",
    claims_pct=f"{weights['Claims']*100:.0f}",
    benefits_pct=f"{weights['Benefits']*100:.0f}",
)

compare_chain = create_stuff_documents_chain(llm, comparison_prompt)

# -------------------------------
# RUN COMPARISON
# -------------------------------
if st.button("🔍 Compare Policies"):
    with st.spinner("Analyzing and comparing..."):
        docs_a = retriever_a.invoke("Summarize key features of this policy")
        docs_b = retriever_b.invoke("Summarize key features of this policy")

        summary_a = extract_chain.invoke({"context": docs_a})
        summary_b = extract_chain.invoke({"context": docs_b})

        combined_doc = Document(
            page_content=f"Policy A:\n{summary_a}\n\nPolicy B:\n{summary_b}"
        )

        result = compare_chain.invoke({"context": [combined_doc]})

    st.subheader("🧩 Comparison Result")
    st.markdown(result)

    st.session_state["comparison_summary"] = result

# -------------------------------
# FOLLOW-UP CHAT
# -------------------------------
st.divider()
st.subheader("💬 Ask Follow-up Questions")

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

if "comparison_summary" in st.session_state:
    followup_chain = RunnableWithMessageHistory(
        compare_chain,
        get_session_history,
        input_messages_key="context",
        history_messages_key="chat_history",
        output_messages_key="output_text"
    )

    question = st.text_input("Ask about the comparison:")
    if question:
        with st.spinner("Thinking..."):
            style_prefix = "Explain in a simple way:\n" if explanation_mode == "Simple" else "Provide a professional analysis:\n"
            question_doc = Document(
                page_content=style_prefix + st.session_state["comparison_summary"] + "\nUser Question: " + question
            )

            response = followup_chain.invoke(
                {"context": [question_doc]},
                config={"configurable": {"session_id": "policy_chat"}}
            )
        st.markdown("**Answer:** " + response)
