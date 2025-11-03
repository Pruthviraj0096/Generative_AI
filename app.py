import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# -------------------------------
# Load .env (optional fallback)
# -------------------------------
load_dotenv()

# -------------------------------
# Sidebar: API Keys
# -------------------------------

st.sidebar.subheader("API Keys")
groq_key = st.sidebar.text_input(
    "Enter GROQ API Key", 
    type="password", 
    value=os.getenv("GROQ_API_KEY") or ""
)
openai_key = st.sidebar.text_input(
    "Enter OpenAI API Key", 
    type="password", 
    value=os.getenv("OPENAI_API_KEY") or ""
)

# -------------------------------
# Load LLM safely
# -------------------------------
try:
    from langchain_groq import ChatGroq
    if groq_key:
        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-8b-instant"
        )
    else:
        raise ValueError("No GROQ key provided")
except (ImportError, ValueError):
    from langchain_community.chat_models import ChatOpenAI
    if openai_key:
        llm = ChatOpenAI(
            openai_api_key=openai_key,
            model_name="gpt-3.5-turbo"
        )
    else:
        st.error("No API key provided for Groq or OpenAI. Please enter a key in the sidebar.")
        st.stop()

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Policy Comparison Advisor", page_icon="📄")
st.title("📄 Policy Advisor Bot")
st.write("Upload **two policy PDFs**, customise what matters most, and get a data-driven recommendation.")

# -------------------------------
# File Uploads
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
# User Weights
# -------------------------------
st.subheader("⚖️ Set Your Priorities")
col1, col2, col3 = st.columns(3)
coverage_weight     = col1.slider("Coverage Importance",        0, 100, 30)
cost_weight         = col2.slider("Cost Importance",            0, 100, 25)
exclusion_weight    = col3.slider("Exclusions Importance",     0, 100, 20)
claims_weight       = col1.slider("Claims Process Importance",  0, 100, 15)
benefits_weight     = col2.slider("Benefits Importance",       0, 100, 10)

total_weight = (coverage_weight + cost_weight + exclusion_weight + claims_weight + benefits_weight)
if total_weight == 0:
    st.warning("Please assign at least one non-zero weight.")
    st.stop()

weights = {
    "Coverage":   coverage_weight   / total_weight,
    "Cost":       cost_weight       / total_weight,
    "Exclusions": exclusion_weight  / total_weight,
    "Claims":     claims_weight     / total_weight,
    "Benefits":   benefits_weight   / total_weight,
}

# -------------------------------
# Explanation Mode
# -------------------------------
st.subheader("🗣️ Choose Explanation Mode")
explanation_mode = st.radio(
    "How should the assistant explain the results?",
    options=["Simple", "Professional"],
    index=0,
    horizontal=True
)

# -------------------------------
# Document Loading + Embedding
# -------------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def load_and_embed(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name

    docs = PyPDFLoader(tmp_path).load()
    os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    splits = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
        model_kwargs={"device": "cpu"}  # Safe on CPU
    )
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever

retriever_a = load_and_embed(policy_a)
retriever_b = load_and_embed(policy_b)

# -------------------------------
# Extraction & Comparison Prompts
# -------------------------------
from langchain_core.prompts import ChatPromptTemplate

extract_prompt = ChatPromptTemplate.from_template("""
You are a policy analyst. Extract the following details from this policy text:

- Coverage details
- Premium / Cost information
- Major exclusions
- Claims process
- Additional or customer benefits

Policy Text:
{context}
""")

def extract_summary(retriever):
    docs = retriever.get_relevant_documents("")
    combined = "\n\n".join([d.page_content for d in docs])
    return llm.generate([{"role": "user", "content": extract_prompt.format(context=combined)}]).generations[0].text

tone = "in a simple and easy-to-understand way" if explanation_mode == "Simple" else "in a detailed and professional tone"

comparison_prompt = ChatPromptTemplate.from_template(f"""
You are an expert insurance policy evaluator.

Use a {tone} explanation style.

Compare **Policy A** and **Policy B** based on these weighted criteria:
- Coverage ({weights['Coverage']*100:.0f}%)
- Cost     ({weights['Cost']*100:.0f}%)
- Exclusions({weights['Exclusions']*100:.0f}%)
- Claims Process({weights['Claims']*100:.0f}%)
- Benefits  ({weights['Benefits']*100:.0f}%)

Below is the extracted information from both policies:
{{context}}

For each criterion:
1. Give a score (1–10) for Policy A and Policy B.
2. Justify briefly.
3. Compute weighted total scores.
4. Recommend which policy is overall better and why (in 3 bullet points).

Structure your response as:

### Comparison Summary Table
| Criteria    | Policy A (Score) | Policy B (Score) | Notes |
|-------------|------------------|------------------|-------|
| …           | …                | …                | …     |

### Weighted Totals
- Policy A Total: X
- Policy B Total: Y

### Recommendation
…
""")

def compare_policies(summary_a: str, summary_b: str):
    combined_doc = f"Policy A:\n{summary_a}\n\nPolicy B:\n{summary_b}"
    resp = llm.generate([{"role": "user", "content": comparison_prompt.format(context=combined_doc)}])
    return resp.generations[0].text

# -------------------------------
# Run Comparison
# -------------------------------
if st.button("🔍 Compare Policies"):
    with st.spinner("Analysing and comparing…"):
        summary_a = extract_summary(retriever_a)
        summary_b = extract_summary(retriever_b)
        result    = compare_policies(summary_a, summary_b)

    st.subheader("🧩 Comparison Result")
    st.markdown(result)
    st.session_state["comparison_summary"] = result

# -------------------------------
# Follow-up Chat
# -------------------------------
st.divider()
st.subheader("💬 Ask Follow-up Questions")

if "store" not in st.session_state:
    st.session_state.store = []

history = st.session_state.store

if "comparison_summary" in st.session_state:
    question = st.text_input("Ask about the comparison:")
    if question:
        with st.spinner("Thinking…"):
            prompt = ("Explain in a simple way:\n" if explanation_mode == "Simple" else "Provide a professional analysis:\n")
            user_input = (prompt + st.session_state["comparison_summary"] + "\nUser Question: " + question)
            resp       = llm.generate([{"role": "user", "content": user_input}])
            answer     = resp.generations[0].text

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            st.session_state.store = history

        st.markdown("**Answer:** " + answer)
