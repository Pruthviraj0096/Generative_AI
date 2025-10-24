import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# INITIAL SETUP
# -------------------------------
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

st.set_page_config(page_title="AI Policy Advisor", page_icon="🤝")
st.title("🤝 AI Policy Advisor")
st.write("Upload **two insurance policies** (PDFs) and get a clear, simple comparison with a final recommendation.")

# -------------------------------
# API SETUP
# -------------------------------
api_key = st.text_input("Enter your Groq API key:", type="password")
if not api_key:
    st.warning("Please enter your Groq API key to continue.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------
# FILE UPLOADS
# -------------------------------
col1, col2 = st.columns(2)
with col1:
    policy_a = st.file_uploader("📄 Upload Policy A", type="pdf", key="policy_a")
with col2:
    policy_b = st.file_uploader("📄 Upload Policy B", type="pdf", key="policy_b")

if not (policy_a and policy_b):
    st.info("Please upload **two PDF files** to start comparison.")
    st.stop()

# -------------------------------
# MODE & PERSONA
# -------------------------------
st.subheader("⚙️ Settings")
mode = st.radio("Choose explanation style:", 
                ["Simple Mode 🧩 (For everyone)", "Advisor Mode 🧠 (Professional tone)"])
persona = st.selectbox("Who are you comparing for?", 
                       ["An individual", "A family", "Senior citizen", "Employee group"])

# -------------------------------
# USER PRIORITIES (Weighting)
# -------------------------------
st.subheader("⚖️ What matters most to you?")
coverage_weight = st.slider("Coverage Importance", 1, 5, 5)
cost_weight = st.slider("Affordability Importance", 1, 5, 3)
claims_weight = st.slider("Claims Process Importance", 1, 5, 4)

user_priorities = f"""
The user's priorities are:
- Coverage importance: {coverage_weight}/5
- Cost importance: {cost_weight}/5
- Claims ease importance: {claims_weight}/5
"""

# -------------------------------
# DOCUMENT LOADING FUNCTION
# -------------------------------
def load_and_embed(pdf_file):
    temp_path = f"./temp_{pdf_file.name}"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.getvalue())
    docs = PyPDFLoader(temp_path).load()
    os.remove(temp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    return Chroma.from_documents(splits, embedding=embeddings).as_retriever()

retriever_a = load_and_embed(policy_a)
retriever_b = load_and_embed(policy_b)

# -------------------------------
# POLICY SUMMARY PROMPT
# -------------------------------
extract_prompt = ChatPromptTemplate.from_template("""
Summarize this insurance policy clearly and concisely.

Include:
- What it covers (in plain English)
- What it doesn’t cover (key exclusions)
- How the claim process works
- Extra benefits or unique features
Keep it short and easy to understand.
Policy Text:
{context}
""")
extract_chain = create_stuff_documents_chain(llm, extract_prompt)

# -------------------------------
# MODE-SPECIFIC INSTRUCTIONS
# -------------------------------
simple_instruction = """
Use very simple, friendly language. 
Avoid insurance jargon. 
If you must use a term (like premium, co-pay, deductible), explain it in brackets.
Use emojis for clarity (✅ good, ⚠️ okay, ❌ not great).
Give short paragraphs and a 'Your Takeaway' summary at the end.
"""

advisor_instruction = """
Use a professional and analytical tone suitable for financial advisors.
Be concise, data-driven, and factual.
Provide structured comparison and a final recommendation with justification.
"""

tone_instruction = simple_instruction if "Simple" in mode else advisor_instruction

# -------------------------------
# COMPARISON PROMPT (MAIN)
# -------------------------------
comparison_prompt = ChatPromptTemplate.from_template(f"""
You are an insurance policy expert explaining things to {persona}.
{tone_instruction}

{user_priorities}

Compare **Policy A** and **Policy B** using these criteria:
- Coverage
- Cost or Premiums
- Exclusions
- Claims process
- Customer benefits

Your output must include the following sections:

1️⃣ **Quick Summary** – Describe each policy briefly.  
2️⃣ **Comparison Table** – Use ✅ (better), ⚠️ (average), ❌ (worse).  
3️⃣ **🏁 Recommendation** – Clearly choose one policy as better overall.  
   - Explicitly name the better policy (Policy A or Policy B).
   - Give 2–3 reasons based on facts.
   - Optionally note: "Policy A is better for families / Policy B suits individuals", etc.  
4️⃣ **💬 Your Takeaway** – Friendly closing summary (max 3 sentences) that a non-expert can understand easily.

Context:
Policy A -> {{"context_a"}}
Policy B -> {{"context_b"}}
""")

compare_chain = create_stuff_documents_chain(llm, comparison_prompt)

# -------------------------------
# RUN COMPARISON
# -------------------------------
if st.button("🔍 Compare Policies"):
    with st.spinner("Analyzing and comparing policies... Please wait..."):
        docs_a = retriever_a.invoke("Summarize key details")
        docs_b = retriever_b.invoke("Summarize key details")

        summary_a = extract_chain.invoke({"context": docs_a})
        summary_b = extract_chain.invoke({"context": docs_b})

        result = compare_chain.invoke({
            "context_a": summary_a["output_text"],
            "context_b": summary_b["output_text"]
        })

    st.subheader("📊 Comparison Result")
    st.markdown(result["output_text"])

# -------------------------------
# OPTIONAL GLOSSARY SECTION
# -------------------------------
st.divider()
st.subheader("🧾 Common Insurance Terms Explained")

INSURANCE_GLOSSARY = {
    "premium": "The amount you pay regularly to keep your policy active.",
    "deductible": "The amount you must pay yourself before insurance starts covering costs.",
    "co-pay": "A percentage of the claim you must pay (e.g., you pay 10%, insurer pays 90%).",
    "sum insured": "The maximum amount you can claim per year.",
    "exclusions": "Situations or expenses that the policy does not cover.",
}

for term, meaning in INSURANCE_GLOSSARY.items():
    with st.expander(term.capitalize()):
        st.write(meaning)
