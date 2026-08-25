import streamlit as st
import json
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres.vectorstores import PGVector
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from sqlalchemy import create_engine, inspect, text
import tempfile
import io
import time
from urllib.parse import quote_plus
import re
from urllib.request import urlopen

# Load environment variables
load_dotenv()

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://m9960-bk.mdindia.com:11434")


def normalize_ollama_base_url(base_url):
    candidate = (base_url or DEFAULT_OLLAMA_BASE_URL).strip().rstrip('/')
    if not candidate:
        return DEFAULT_OLLAMA_BASE_URL
    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"
    return candidate


def is_ollama_reachable(base_url):
    try:
        url = f"{normalize_ollama_base_url(base_url)}/api/tags"
        with urlopen(url, timeout=2) as response:
            return response.status < 400
    except Exception:
        return False


# Page Config
st.set_page_config(
    page_title="Medical Coding Intelligence",
    page_icon="🧬",
    layout="wide",
)

# Database Configuration
DB_USER = os.getenv("DB_USER", "devread_write")
DB_PASSWORD = os.getenv("DB_PASSWORD", "devread_write@123")
DB_HOST = os.getenv("DB_HOST", "devdb.mdindia.com")
DB_PORT = os.getenv("DB_PORT", "5000")
DB_NAME = os.getenv("DB_NAME", "apachedbdev")
DB_SCHEMA = os.getenv("DB_SCHEMA", "icd")

def get_connection_string(db_name=None):
    user = os.getenv("DB_USER", DB_USER)
    # URL-encode password to handle special characters like '@'
    pw = quote_plus(os.getenv("DB_PASSWORD", DB_PASSWORD))
    host = os.getenv("DB_HOST", DB_HOST)
    port = os.getenv("DB_PORT", DB_PORT)
    db = db_name if db_name else DB_NAME

    return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"


def get_column_names(engine, schema_name, object_name):
    """Return column names without reflecting PostgreSQL-specific data types."""
    query = text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = :schema_name AND table_name = :object_name "
        "ORDER BY ordinal_position"
    )
    with engine.connect() as conn:
        return list(conn.execute(query, {
            "schema_name": schema_name,
            "object_name": object_name,
        }).scalars())

CONNECTION_STRING = get_connection_string()
COLLECTION_NAME = "medical_codes"

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stTextArea textarea {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
    .stButton button {
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .stExpander {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        margin-bottom: 1rem !important;
    }
    .highlight {
        color: #58a6ff;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper Functions
def get_embeddings_model(source, model_name, api_key=None, ollama_base_url=None):
    if source == "Ollama":
        base_url = normalize_ollama_base_url(ollama_base_url)
        if not is_ollama_reachable(base_url):
            base_url = DEFAULT_OLLAMA_BASE_URL
        return OllamaEmbeddings(model=model_name, base_url=base_url)
    else:
        return OpenAIEmbeddings(openai_api_key=api_key, model=model_name)

def get_chat_model(source, model_name, api_key=None, ollama_base_url=None):
    if source == "Ollama":
        chat_model = "llama3" if "embed" in model_name else model_name
        return ChatOllama(model=chat_model, temperature=0, base_url=ollama_base_url)
    else:
        return ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0)


def embed_documents_resilient(model, texts, max_batch_size=8, retries=3):
    """Embed documents in small batches and tolerate transient Ollama runner restarts."""
    texts = list(texts)
    if not texts:
        return []

    vectors = []
    for start in range(0, len(texts), max_batch_size):
        batch = texts[start:start + max_batch_size]
        for attempt in range(retries):
            try:
                vectors.extend(model.embed_documents(batch))
                break
            except Exception:
                if attempt == retries - 1:
                    if len(batch) == 1:
                        raise
                    midpoint = len(batch) // 2
                    vectors.extend(embed_documents_resilient(model, batch[:midpoint], max_batch_size, retries))
                    vectors.extend(embed_documents_resilient(model, batch[midpoint:], max_batch_size, retries))
                    break
                time.sleep(1.5 * (attempt + 1))
    return vectors


def get_storage_config_from_state():
    storage_db = st.session_state.get('storage_db_name', DB_NAME)
    storage_schema = st.session_state.get('storage_schema_name', DB_SCHEMA)
    storage_table = st.session_state.get('storage_table_name')
    storage_text_column = st.session_state.get('storage_text_column', 'content')
    storage_vector_column = st.session_state.get('storage_vector_column', 'embedding')
    storage_key_column = st.session_state.get('storage_key_column')

    # This target requires its source bill ID on every inserted embedding row.
    if storage_table == 'ihx_bill_embeddings':
        storage_key_column = 'ihx_billid'

    if not storage_table:
        return None

    return {
        "db": storage_db,
        "schema": storage_schema,
        "table": storage_table,
        "text_column": storage_text_column,
        "vector_column": storage_vector_column,
        "target_key_column": storage_key_column,
        "same_table": (
            st.session_state.get('ingest_db') == storage_db and
            st.session_state.get('ingest_schema') == storage_schema and
            st.session_state.get('ingest_table') == storage_table and
            bool(st.session_state.get('ingest_key_column'))
        ),
        "key_column": st.session_state.get('ingest_key_column')
    }


def store_embedding_batch(batch_texts, batch_vectors, batch_row_ids, storage_config):
    if not storage_config:
        return

    engine = create_engine(get_connection_string(storage_config['db']))
    with engine.connect() as conn:
        if storage_config['same_table'] and storage_config['key_column']:
            update_query = text(
                f'UPDATE "{storage_config["schema"]}"."{storage_config["table"]}" '
                f'SET "{storage_config["vector_column"]}" = :embedding '
                f'WHERE "{storage_config["key_column"]}" = :row_id'
            )
            for row_id, vec_raw in zip(batch_row_ids, batch_vectors):
                conn.execute(update_query, {
                    "embedding": str(list(vec_raw)),
                    "row_id": row_id
                })
        else:
            target_key = storage_config.get('target_key_column')
            if target_key:
                if any(row_id is None for row_id in batch_row_ids):
                    raise ValueError(
                        f'Target key column "{target_key}" requires a source row ID for every embedding. '
                        'Select the source bill ID as the Row Key Column and fetch the data again.'
                    )
                insert_query = text(
                    f'INSERT INTO "{storage_config["schema"]}"."{storage_config["table"]}" '
                    f'("{target_key}", "{storage_config["vector_column"]}") '
                    f'VALUES (:row_id, :embedding) '
                    f'ON CONFLICT ("{target_key}") DO UPDATE SET '
                    f'"{storage_config["vector_column"]}" = EXCLUDED."{storage_config["vector_column"]}"'
                )
                for row_id, vec_raw in zip(batch_row_ids, batch_vectors):
                    conn.execute(insert_query, {
                        "row_id": row_id,
                        "embedding": str(list(vec_raw))
                    })
            else:
                insert_query = text(
                    f'INSERT INTO "{storage_config["schema"]}"."{storage_config["table"]}" '
                    f'("{storage_config["vector_column"]}") VALUES (:embedding)'
                )
                for vec_raw in batch_vectors:
                    conn.execute(insert_query, {"embedding": str(list(vec_raw))})
        conn.commit()


def get_vectorstore(embeddings_model, connection=None, collection=None):
    return PGVector(
        embeddings=embeddings_model,
        collection_name=collection if collection else COLLECTION_NAME,
        connection=connection if connection else CONNECTION_STRING,
        use_jsonb=True,
    )

def parse_vector_string(val):
    """Robustly parse many vector string formats (standard JSON, quoted numbers with commas, etc.)."""
    if not val or not isinstance(val, (str, list, np.ndarray)):
        return None
    
    if isinstance(val, (list, np.ndarray)):
        return np.array(val)
        
    try:
        # 1. Try standard JSON parsing first
        return np.array(json.loads(val))
    except:
        # 2. Fallback: Extract all numbers (including negatives and decimals) using regex
        # This handles formats like: '-0.0001,' '-0.0002,' etc.
        numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", val)
        if numbers:
            return np.array([float(n) for n in numbers])
    return None

def flatten_json_to_texts(data):
    """Convert the specific JSON structure into a list of descriptive strings for embedding."""
    texts = []
    
    # Input part
    inp = data.get("input", {})
    if inp:
        texts.append(f"ICD10 Code: {inp.get('icd10cm_code', '')}. Description: {inp.get('description', '')}")
    
    # Mappings part
    mappings = data.get("relevant_mappings", {}).get("cpt_codes", [])
    for cpt in mappings:
        texts.append(f"CPT Code: {cpt.get('code', '')}. Description: {cpt.get('description', '')}. Why relevant: {cpt.get('why_relevant', '')}")
    
    return texts

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    embedding_source = st.radio("Select Embedding Source", ["Ollama", "OpenAI"])
    
    if embedding_source == "Ollama":
        ollama_model = st.text_input("Ollama Model Name", value="nomic-embed-text")
        ollama_base_url = st.text_input("Ollama Server URL", value=DEFAULT_OLLAMA_BASE_URL, key="ollama_base_url")
        ollama_base_url = normalize_ollama_base_url(ollama_base_url)
        if not is_ollama_reachable(ollama_base_url):
            st.warning(f"Ollama endpoint {ollama_base_url} is not reachable. Falling back to {DEFAULT_OLLAMA_BASE_URL}.")
            ollama_base_url = DEFAULT_OLLAMA_BASE_URL
    else:
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        openai_model = st.selectbox("OpenAI Model", ["text-embedding-3-small", "text-embedding-3-large"])
        ollama_base_url = None

# Main UI
st.title("🚀 Medical Coding Intelligence")
st.markdown("Bridge Medical Codes and Patient Records using Vector Search.")

# Tabs for different operations
tab1, tab2 = st.tabs(["📄 Code Ingestion", "🔍 PDF Analysis"])

with tab1:
    # Default JSON for reference
    default_json = {
      "input": {
        "icd10cm_code": "R10.33",
        "description": "Periumbilical pain"
      },
      "relevant_mappings": {
        "cpt_codes": [
          {
            "code": "99284",
            "description": "Emergency department visit, moderate complexity",
            "why_relevant": "Common E/M level for ED evaluation of acute periumbilical pain with diagnostic workup.",
            "confidence": 0.9
          }
        ]
      }
    }

    input_method = st.radio("Select Input Method", ["Manual Input", "Excel Upload", "PDF Upload", "PostgreSQL Table"], horizontal=True)
    st.session_state['input_method'] = input_method
    
    json_input = ""
    to_process = []
    
    if input_method == "Manual Input":
        json_input = st.text_area("JSON Input", value=json.dumps(default_json, indent=2), height=200)
    elif input_method == "Excel Upload":
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.success(f"📂 Found **{len(df)}** rows.")
            col1, col2 = st.columns(2)
            with col1:
                column_name = st.selectbox("Select Column Containing JSON/Text", df.columns)
            with col2:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=len(df), value=min(20, len(df)))
            
            if column_name:
                for val in df[column_name]:
                    if pd.isna(val): continue
                    try:
                        parsed = json.loads(val) if isinstance(val, str) and (val.strip().startswith('{') or val.strip().startswith('[')) else val
                        to_process.append(parsed)
                    except:
                        to_process.append(str(val))
    elif input_method == "PDF Upload":
        uploaded_pdf = st.file_uploader("Upload PDF File for Ingestion", type=["pdf"])
        if uploaded_pdf:
            st.success(f"📂 PDF file uploaded: **{uploaded_pdf.name}**")
            pdf_mode = st.radio("Ingestion Mode", ["Split into Chunks", "Single Document"], horizontal=True)
            batch_size = st.number_input("Batch Size (Chunks/Docs)", min_value=1, value=10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.getvalue())
                tmp_path = tmp.name
            try:
                all_text = ""
                with pdfplumber.open(tmp_path) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted: all_text += extracted + "\n"
                
                if all_text.strip():
                    if pdf_mode == "Split into Chunks":
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                        to_process = text_splitter.split_text(all_text)
                    else:
                        to_process = [all_text]
                else:
                    st.error("No text found in PDF.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(tmp_path): os.unlink(tmp_path)
    elif input_method == "PostgreSQL Table":
        try:
            col_db1, col_db2 = st.columns(2)
            with col_db1:
                ingest_db = st.text_input("Database Name", value="apachedbdev", key="ingest_db_name").strip()
            with col_db2:
                ingest_schema = st.text_input("Schema Name", value="icd", key="ingest_schema_name").strip()
            
            available_objects = []
            if ingest_db and ingest_schema:
                state_key = f"ingest_objects_{ingest_db}_{ingest_schema}"
                refresh_tables = st.button("Refresh Tables/Views", key="refresh_ingest_tables")

                try:
                    if refresh_tables or state_key not in st.session_state:
                        engine = create_engine(get_connection_string(ingest_db))
                        inspector = inspect(engine)
                        tables = inspector.get_table_names(schema=ingest_schema)
                        views = inspector.get_view_names(schema=ingest_schema)
                        available_objects = list(dict.fromkeys(tables + views))
                        st.session_state[state_key] = available_objects
                    else:
                        available_objects = st.session_state.get(state_key, [])
                except Exception as e:
                    st.error(f"Connection Error: {e}")
                    available_objects = []

            if ingest_db and ingest_schema and available_objects:
                st.info(f"🔗 Connected to `{ingest_db}` | Schema: `{ingest_schema}`")
                col_db1_2, col_db2_2, col_db3_2, col_db4_2 = st.columns(4)
                
                with col_db1_2:
                    selected_table = st.selectbox("Select Table/View", available_objects, key="ingest_table_sel")
                
                if selected_table:
                    engine = create_engine(get_connection_string(ingest_db))
                    inspector = inspect(engine)
                    cols = get_column_names(engine, ingest_schema, selected_table)
                    with col_db2_2:
                        selected_column = st.selectbox("Select Column", cols, key="ingest_col_sel")
                    with col_db3_2:
                        selected_key_column = st.selectbox("Select Row Key Column", cols, key="ingest_key_col_sel")
                    with col_db4_2:
                        db_batch_size = st.number_input("Batch Size", min_value=1, value=50, key="ingest_batch_size")
                        batch_size = db_batch_size
                    
                    state_key = f"last_config_{ingest_db}_{ingest_schema}"
                    current_config = f"{selected_table}_{selected_column}_{selected_key_column}"
                    if state_key not in st.session_state or st.session_state[state_key] != current_config:
                        st.session_state['db_to_process'] = []
                        st.session_state[state_key] = current_config

                if st.button("🚀 Fetch Data from Table/View"):
                    with st.spinner(f"Fetching from `{ingest_schema}`.`{selected_table}`..."):
                        with engine.connect() as conn:
                            query = text(f'SELECT "{selected_key_column}", "{selected_column}" FROM "{ingest_schema}"."{selected_table}" WHERE "{selected_column}" IS NOT NULL')
                            result = conn.execute(query)
                            for row in result:
                                row_id, val = row
                                if val is not None:
                                    to_process.append((row_id, val))
                        st.success(f"📥 Fetched **{len(to_process)}** rows.")
                        if to_process:
                            st.markdown("#### Top 5 fetched rows")
                            preview_df = pd.DataFrame(
                                to_process[:5],
                                columns=[selected_key_column, selected_column]
                            )
                            st.dataframe(preview_df, use_container_width=True)
                        st.session_state['db_to_process'] = to_process
                        st.session_state['ingest_db'] = ingest_db
                        st.session_state['ingest_schema'] = ingest_schema
                        st.session_state['ingest_table'] = selected_table
                        st.session_state['ingest_value_column'] = selected_column
                        st.session_state['ingest_key_column'] = selected_key_column
            elif ingest_db and ingest_schema:
                st.warning("No tables or views found for this schema. Please verify the schema name and click Refresh Tables/Views.")
        except Exception as e:
            st.error(f"Connection Error: {e}")

    # Restore to_process from session state for DB method
    if input_method == "PostgreSQL Table" and 'db_to_process' in st.session_state:
        to_process = st.session_state['db_to_process']
    
    if to_process:
        st.info(f"📍 **Ready to embed {len(to_process)} records** from {input_method}.")

    st.markdown("---")
    st.subheader("💾 Storage Configuration")
    col_s1, col_s2, col_s3 = st.columns(3)
    default_storage_db = st.session_state.get('ingest_db', 'apachedbdev') if input_method == 'PostgreSQL Table' else 'apachedbdev'
    default_storage_schema = st.session_state.get('ingest_schema', 'icd') if input_method == 'PostgreSQL Table' else 'icd'

    with col_s1:
        storage_db = st.text_input("Storage Database Name", value=default_storage_db, key="storage_db_name")
    with col_s2:
        storage_schema = st.text_input("Storage Schema Name", value=default_storage_schema, key="storage_schema_name")

    storage_table = None
    storage_text_column = "content"
    storage_vector_column = "embedding"
    storage_key_column = None
    storage_column_options = []
    storage_table_names = []
    storage_view_names = []
    storage_object_names = []
    storage_same_table = False

    if storage_db and storage_schema:
        try:
            storage_engine = create_engine(get_connection_string(storage_db))
            storage_inspector = inspect(storage_engine)
            storage_table_names = storage_inspector.get_table_names(schema=storage_schema)
            storage_view_names = storage_inspector.get_view_names(schema=storage_schema)
            storage_object_names = list(dict.fromkeys(storage_table_names + storage_view_names))

            if storage_object_names:
                default_table = st.session_state.get('ingest_table', 'documents') if input_method == 'PostgreSQL Table' else 'documents'
                default_index = storage_object_names.index(default_table) if default_table in storage_object_names else (storage_object_names.index('documents') if 'documents' in storage_object_names else 0)
                with col_s3:
                    storage_table = st.selectbox(
                        "Collection/Table/View Name",
                        storage_object_names,
                        index=default_index,
                        key="storage_table_name"
                    )
            else:
                with col_s3:
                    storage_table = st.text_input("Collection/Table/View Name", value="documents", key="storage_table_name")
                st.warning(f"No tables or views found in schema `{storage_schema}`.")

            if storage_table and storage_table in storage_object_names:
                storage_column_options = get_column_names(storage_engine, storage_schema, storage_table)
                if storage_column_options:
                    if (
                        input_method == 'PostgreSQL Table'
                        and storage_db == st.session_state.get('ingest_db')
                        and storage_schema == st.session_state.get('ingest_schema')
                        and storage_table == st.session_state.get('ingest_table')
                    ):
                        storage_same_table = True
                        default_vector_column = (
                            'final_json_vectors' if 'final_json_vectors' in storage_column_options
                            else 'embedding' if 'embedding' in storage_column_options
                            else storage_column_options[0]
                        )
                        storage_vector_column = st.selectbox(
                            "Vector Column",
                            storage_column_options,
                            index=storage_column_options.index(default_vector_column),
                            key="storage_vector_column"
                        )
                        st.markdown(f"Selected same source object `{storage_table}`. Embeddings will be updated using key column `{st.session_state.get('ingest_key_column')}`.")
                    else:
                        if input_method == 'PostgreSQL Table':
                            col_key, col_vector = st.columns(2)
                            source_key = st.session_state.get('ingest_key_column')
                            default_key = source_key if source_key in storage_column_options else ('ihx_billid' if 'ihx_billid' in storage_column_options else storage_column_options[0])
                            with col_key:
                                storage_key_column = st.selectbox(
                                    "Target Row Key Column",
                                    storage_column_options,
                                    index=storage_column_options.index(default_key),
                                    key="storage_key_column"
                                )
                            with col_vector:
                                storage_vector_column = st.selectbox(
                                    "Vector Column",
                                    storage_column_options,
                                    index=storage_column_options.index('embedding') if 'embedding' in storage_column_options else 0,
                                    key="storage_vector_column"
                                )
                            st.caption("The source row key and generated embedding will be inserted; other nullable columns are omitted.")
                        else:
                            storage_vector_column = st.selectbox(
                                "Vector Column",
                                storage_column_options,
                                index=storage_column_options.index('embedding') if 'embedding' in storage_column_options else 0,
                                key="storage_vector_column"
                            )
                            st.caption("Only the generated embedding will be inserted into this target object.")
                else:
                    st.warning(f"No columns found for object `{storage_table}`.")
            elif storage_table:
                st.warning(f"Object `{storage_table}` not found in schema `{storage_schema}`.")
        except Exception as e:
            st.error(f"Could not inspect storage table: {e}")
    
    if st.button("Generate & Preview Vectors"):
        if input_method == "Manual Input" and json_input:
            try:
                to_process = [json.loads(json_input)]
            except:
                st.error("Invalid JSON.")
                to_process = []

        if to_process:
            if embedding_source == "OpenAI" and not api_key:
                st.error("Please provide an OpenAI API Key in the sidebar.")
            else:
                with st.spinner("Processing in batches..."):
                    model = get_embeddings_model(
                        embedding_source,
                        ollama_model if embedding_source == "Ollama" else openai_model,
                        api_key if embedding_source == "OpenAI" else None,
                        ollama_base_url if embedding_source == "Ollama" else None,
                    )
                    
                    storage_config = get_storage_config_from_state()
                    preview_texts = []
                    preview_vectors = []
                    all_row_ids = []
                    
                    # Prepare texts and row IDs
                    expanded_data = []
                    for i, item in enumerate(to_process):
                        row_id = None
                        if isinstance(item, tuple) and len(item) == 2:
                            row_id, item = item

                        if isinstance(item, (dict, list)):
                            texts = flatten_json_to_texts(item)
                            for t in texts:
                                expanded_data.append((t, {"source_index": i, "content_type": "coding_logic"}, row_id))
                        else:
                            expanded_data.append((str(item), {"source_index": i, "content_type": "coding_logic"}, row_id))

                    # Batch process embeddings and stream them into the selected Postgres target
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    actual_batch_size = batch_size if input_method != "Manual Input" else 1
                    total_items = len(expanded_data)
                    
                    for i in range(0, total_items, actual_batch_size):
                        batch = expanded_data[i: i + actual_batch_size]
                        batch_texts = [b[0] for b in batch]
                        batch_metas = [b[1] for b in batch]
                        batch_row_ids = [b[2] for b in batch]
                        
                        vectors = embed_documents_resilient(model, batch_texts)

                        # Generate first; persist exactly once through the explicit Save button.
                        preview_texts.extend(batch_texts)
                        preview_vectors.extend(vectors)
                        all_row_ids.extend(batch_row_ids)
                        
                        progress = min((i + actual_batch_size) / total_items, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Processed {min(i + actual_batch_size, total_items)}/{total_items} items...")

                    st.session_state['pending_texts'] = preview_texts
                    st.session_state['pending_metadatas'] = [{"source_index": 0, "content_type": "coding_logic"}] * len(preview_texts)
                    st.session_state['pending_vectors'] = preview_vectors
                    st.session_state['pending_row_ids'] = all_row_ids
                    
                    if storage_config:
                        st.info("✅ Embeddings are ready. Click Save to write them to PostgreSQL.")
                    if len(preview_vectors) > 5:
                        st.info(f"Showing first 5 of {len(preview_vectors)} vectors.")

    # --- Stable UI Section (Outside of Generate button) ---
    if 'pending_texts' in st.session_state and st.session_state['pending_texts']:
        st.markdown("---")
        
        all_texts = st.session_state['pending_texts']
        all_vectors = st.session_state['pending_vectors']
        
        st.success(f"✅ Generated {len(all_vectors)} vectors.")
        
        # ----- Download Button -----
        df_download = pd.DataFrame({
            "id": ["Auto"] * len(all_texts),
            "content": all_texts,
            "embedding": [json.dumps(v) for v in all_vectors]
        })
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df_download.to_excel(writer, index=False, sheet_name='Embeddings')
        
        st.download_button(
            label="📥 Download Embeddings as XLSX",
            data=buffer.getvalue(),
            file_name="embeddings_export.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.subheader("🔢 Vector Preview")
        
        # Display Preview aligned with DB structure
        preview_df = pd.DataFrame({
            "id": ["Auto"] * len(all_texts),
            "content": all_texts,
            "embedding": [str(v[:10])[:-1] + ", ...]" for v in all_vectors]
        })
        st.table(preview_df.head(5))
        
        if len(all_vectors) > 5:
            st.info(f"Showing first 5 of {len(all_vectors)} vectors.")

        st.markdown("---")

        if st.button("💾 Save to Vector Store (PostgreSQL)"):
            if embedding_source == "OpenAI" and not api_key:
                st.error("Please provide an OpenAI API Key in the sidebar.")
            elif not storage_table:
                st.error("Please select a storage table first.")
            elif not storage_column_options:
                st.error("Please select a valid table with columns.")
            else:
                with st.spinner(f"Storing embeddings in `{storage_db}`.`{storage_schema}`.`{storage_table}`..."):
                    try:
                        total_to_store = len(st.session_state['pending_vectors'])
                        storage_progress_bar = st.progress(0)
                        storage_status_text = st.empty()
                        engine = create_engine(get_connection_string(storage_db))
                        with engine.connect() as conn:
                            if input_method == 'PostgreSQL Table' and storage_same_table and 'pending_row_ids' in st.session_state:
                                key_column = st.session_state.get('ingest_key_column')
                                update_query = text(
                                    f'UPDATE "{storage_schema}"."{storage_table}" SET "{storage_vector_column}" = :embedding WHERE "{key_column}" = :row_id'
                                )
                                for i, vec_raw in enumerate(st.session_state['pending_vectors']):
                                    row_id = st.session_state['pending_row_ids'][i]
                                    vec_str = str(list(vec_raw))
                                    conn.execute(update_query, {"embedding": vec_str, "row_id": row_id})
                                    stored_count = i + 1
                                    storage_progress_bar.progress(stored_count / total_to_store)
                                    storage_status_text.text(f"Stored {stored_count}/{total_to_store} embeddings...")
                            else:
                                effective_storage_key = (
                                    'ihx_billid' if storage_table == 'ihx_bill_embeddings'
                                    else storage_key_column
                                )
                                include_key = (
                                    input_method == 'PostgreSQL Table'
                                    and effective_storage_key
                                    and 'pending_row_ids' in st.session_state
                                )
                                if include_key:
                                    insert_query = text(
                                        f'INSERT INTO "{storage_schema}"."{storage_table}" '
                                        f'("{effective_storage_key}", "{storage_vector_column}") '
                                        f'VALUES (:row_id, :embedding) '
                                        f'ON CONFLICT ("{effective_storage_key}") DO UPDATE SET '
                                        f'"{storage_vector_column}" = EXCLUDED."{storage_vector_column}"'
                                    )
                                else:
                                    insert_query = text(
                                        f'INSERT INTO "{storage_schema}"."{storage_table}" '
                                        f'("{storage_vector_column}") VALUES (:embedding)'
                                    )
                                for i, vec_raw in enumerate(st.session_state['pending_vectors']):
                                    parameters = {"embedding": str(list(vec_raw))}
                                    if include_key:
                                        parameters["row_id"] = st.session_state['pending_row_ids'][i]
                                    conn.execute(insert_query, parameters)
                                    stored_count = i + 1
                                    storage_progress_bar.progress(stored_count / total_to_store)
                                    storage_status_text.text(f"Stored {stored_count}/{total_to_store} embeddings...")
                            conn.commit()
                        storage_status_text.text(f"Stored {total_to_store}/{total_to_store} embeddings.")
                        st.success(f"✅ Successfully stored {len(st.session_state['pending_texts'])} vectors in `{storage_schema}`.`{storage_table}` table!")
                    except Exception as e:
                        st.error(f"SQL Error: {e}")
                        if storage_column_options:
                            st.info(f"Ensure the table exists with '{storage_vector_column}' column and, if updating, key column '{st.session_state.get('ingest_key_column')}'.")
                        else:
                            st.info("Ensure the table exists with a valid vector column.")

                    del st.session_state['pending_texts']
                    del st.session_state['pending_metadatas']
                    if 'pending_vectors' in st.session_state:
                        del st.session_state['pending_vectors']
                    if 'pending_row_ids' in st.session_state:
                        del st.session_state['pending_row_ids']

with tab2:
    st.info("💡 **Transient Analysis**: Content from uploaded PDFs is processed in-memory and is **never saved** to your database or vector store.")
    pdf_file = st.file_uploader("Upload Medical PDF for Analysis", type="pdf")
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            tmp_path = tmp.name
        
        st.markdown("### 🔍 Search Configuration")
        col_m1, col_m2 = st.columns([1, 2])
        with col_m1:
            analysis_mode = st.radio("Analysis Mode", ["Split into Chunks", "Single Document"], horizontal=False)
            if 'last_analysis_mode' not in st.session_state:
                st.session_state['last_analysis_mode'] = analysis_mode
            
            # Reset chunks/vectors if mode changes
            if st.session_state['last_analysis_mode'] != analysis_mode:
                if 'pdf_chunks' in st.session_state: del st.session_state['pdf_chunks']
                if 'pdf_vectors' in st.session_state: del st.session_state['pdf_vectors']
                st.session_state['last_analysis_mode'] = analysis_mode

        with col_m2:
            search_target = st.radio("Search Target", ["Default Vector Store (medical_codes)", "Custom PostgreSQL Table"], horizontal=True)
        
        custom_search_config = {}
        if search_target == "Custom PostgreSQL Table":
            try:
                # Database and Schema selection
                col_db_sel1, col_db_sel2 = st.columns(2)
                with col_db_sel1:
                    target_db = st.text_input("Database Name", value="apachedbdev", key="pdftab_db_name")
                with col_db_sel2:
                    target_schema = st.text_input("Schema Name", value="icd", key="pdftab_schema_name")
                
                if target_db:
                    engine = create_engine(get_connection_string(target_db))
                    inspector = inspect(engine)
                    
                    # Fetch tables for the specific schema
                    tables = inspector.get_table_names(schema=target_schema)
                    
                    col_t_sel1, col_t_sel2 = st.columns([2, 1])
                    with col_t_sel1:
                        target_table = st.selectbox("Select Target Table", tables, key="pdftab_table_ref_v2")
                    
                    if target_table:
                        cols = get_column_names(engine, target_schema, target_table)
                        col_c1, col_c2 = st.columns(2)
                        with col_c1:
                            target_column = st.selectbox("Select Text/Description Column", cols, key="pdftab_col_ref_v2")
                        with col_c2:
                            vector_column = st.selectbox("Select Vector Column (Optional)", ["None"] + cols, key="pdftab_vec_ref")
                        
                        custom_search_config = {
                            "database": target_db,
                            "schema": target_schema,
                            "table": target_table,
                            "column": target_column,
                            "vector_column": vector_column if vector_column != "None" else None,
                            "engine": engine
                        }
            except Exception as e:
                st.error(f"Database Error: {e}")

        # 1. Extraction step
        if st.button("🔍 Extract & Preview PDF Text"):
            with st.spinner(f"Extracting text in **{analysis_mode}** mode..."):
                all_text = ""
                try:
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            if text:
                                all_text += text + "\n"
                    
                    if not all_text.strip():
                        st.error("No text could be extracted from this PDF. It might be an image-only PDF.")
                    else:
                        if analysis_mode == "Split into Chunks":
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                            chunks = text_splitter.split_text(all_text)
                        else:
                            chunks = [all_text] # Single document mode
                        
                        st.session_state['pdf_chunks'] = chunks
                        # Clear old vectors since content changed
                        if 'pdf_vectors' in st.session_state: del st.session_state['pdf_vectors']
                        
                        st.success(f"Extracted **{len(chunks)}** unit(s) from the PDF.")
                except Exception as e:
                    st.error(f"Error during PDF extraction: {e}")
        
        # Preview extracted chunks
        if 'pdf_chunks' in st.session_state:
            st.subheader("📝 Extracted PDF Chunks")
            for i, chunk_text in enumerate(st.session_state['pdf_chunks'][:5]):
                st.info(f"Chunk {i+1}: {chunk_text[:200]}...")
            if len(st.session_state['pdf_chunks']) > 5:
                st.write(f"... and {len(st.session_state['pdf_chunks']) - 5} more chunks.")

            # New: Preview PDF Vectors
            if st.button("🔢 Generate & Preview PDF Vectors"):
                if embedding_source == "OpenAI" and not api_key:
                    st.error("Please provide an OpenAI API Key in the sidebar.")
                else:
                    with st.spinner("Generating embeddings for PDF chunks..."):
                        model = get_embeddings_model(
                            embedding_source,
                            ollama_model if embedding_source == "Ollama" else openai_model,
                            api_key if embedding_source == "OpenAI" else None,
                            ollama_base_url if embedding_source == "Ollama" else None,
                        )
                        pdf_vectors = embed_documents_resilient(model, st.session_state['pdf_chunks'])
                        st.session_state['pdf_vectors'] = pdf_vectors
                        st.success(f"Generated {len(pdf_vectors)} vectors for PDF chunks.")
                        
                        st.subheader("🔢 PDF Vector Preview")
                        for i, vec in enumerate(pdf_vectors[:5]):
                            with st.expander(f"Chunk Vector {i+1} (Size: {len(vec)})"):
                                st.markdown(f"**Content**: {st.session_state['pdf_chunks'][i][:200]}...")
                                st.code(str(vec[:10])[:-1] + ", ...]", language="python")
                                st.json(vec)

            # 2. Search step
            if st.button("🚀 Run Semantic Search on PDF"):
                if embedding_source == "OpenAI" and not api_key:
                    st.error("Please provide an OpenAI API Key in the sidebar.")
                elif search_target == "Custom PostgreSQL Table" and not custom_search_config:
                    st.error("Please select a target table and column.")
                else:
                    with st.spinner("Searching..."):
                        # Setup Model
                        model = get_embeddings_model(
                            embedding_source,
                            ollama_model if embedding_source == "Ollama" else openai_model,
                            api_key if embedding_source == "OpenAI" else None,
                            ollama_base_url if embedding_source == "Ollama" else None,
                        )

                        if 'pdf_vectors' in st.session_state:
                            pdf_vectors = st.session_state['pdf_vectors']
                        else:
                            pdf_vectors = embed_documents_resilient(model, st.session_state['pdf_chunks'])
                            st.session_state['pdf_vectors'] = pdf_vectors

                        st.subheader("💡 Semantic Analysis Results")
                        matches_found = False
                        all_match_data = []

                        if search_target == "Default Vector Store (medical_codes)":
                            try:
                                vectorstore = get_vectorstore(model)
                                for idx, chunk_content in enumerate(st.session_state['pdf_chunks']):
                                    # Use the pre-calculated vector for search
                                    chunk_vector = pdf_vectors[idx]
                                    results = vectorstore.similarity_search_with_score_by_vector(chunk_vector, k=1)
                                    if results and results[0][1] < 1.5: 
                                        matches_found = True
                                        doc, score = results[0]
                                        
                                        all_match_data.append({
                                            "pdf_text": chunk_content,
                                            "db_match": doc.page_content,
                                            "score": score
                                        })
                            except Exception as e:
                                st.error(f"Vector Store Error: {e}")
                                if "InsufficientPrivilege" in str(e):
                                    st.warning("⚠️ **Permission Hint**: Your database user cannot manage extensions. Use 'Custom PostgreSQL Table' or contact admin.")
                        else:
                            # Custom table search
                            with custom_search_config["engine"].connect() as conn:
                                cols_to_fetch = [custom_search_config['column']]
                                if custom_search_config['vector_column']:
                                    cols_to_fetch.append(custom_search_config['vector_column'])
                                
                                # Use schema qualification
                                schema = custom_search_config['schema']
                                table = custom_search_config['table']
                                col = custom_search_config['column']
                                
                                query_str = f'SELECT * FROM "{schema}"."{table}" WHERE "{col}" IS NOT NULL'
                                
                                # If vector column is selected, ensure it's also not null
                                if custom_search_config['vector_column']:
                                    query_str += f' AND "{custom_search_config["vector_column"]}" IS NOT NULL'
                                
                                query = text(query_str)
                                result_proxy = conn.execute(query)
                                db_results = result_proxy.fetchall()
                                column_names = list(result_proxy.keys())
                                
                                db_rows = []
                                db_vectors = []
                                
                                vector_col_name = custom_search_config['vector_column']
                                text_col_name = custom_search_config['column']

                                for row in db_results:
                                    # Create a dictionary of the full row
                                    row_dict = dict(zip(column_names, row))
                                    
                                    # Ensure we have the text for display/fallback
                                    if not row_dict.get(text_col_name): continue
                                    
                                    db_rows.append(row_dict)
                                    
                                    if vector_col_name and row_dict.get(vector_col_name):
                                        vec = parse_vector_string(row_dict[vector_col_name])
                                        db_vectors.append(vec)
                                    else:
                                        db_vectors.append(None)

                            if not db_rows:
                                st.warning("No data found in selected table/column.")
                            else:
                                # REMOVED: In-session embedding of database items
                                valid_db_indices = [i for i, v in enumerate(db_vectors) if v is not None]
                                
                                if not valid_db_indices:
                                    st.error("No valid pre-computed vectors found in the selected column. Please ensure the column contains vector data (e.g., '[0.1, 0.2, ...]').")
                                else:
                                    st.success(f"Using {len(valid_db_indices)} pre-computed vectors from database.")

                                    for idx, chunk_content in enumerate(st.session_state['pdf_chunks']):
                                        chunk_vector = pdf_vectors[idx]
                                        
                                        best_score = 2.0 
                                        best_row = None
                                        
                                        for i in valid_db_indices:
                                            db_vec = db_vectors[i]
                                            
                                            dot_prod = np.dot(chunk_vector, db_vec)
                                            norm_a = np.linalg.norm(chunk_vector)
                                            norm_b = np.linalg.norm(db_vec)
                                            
                                            if norm_a == 0 or norm_b == 0: continue
                                            
                                            cosine_sim = dot_prod / (norm_a * norm_b)
                                            distance = 1 - cosine_sim
                                            
                                            if distance < best_score:
                                                best_score = distance
                                                best_row = db_rows[i]
                                        
                                        if best_score < 0.5: 
                                            matches_found = True
                                            all_match_data.append({
                                                "pdf_text": chunk_content,
                                                "db_match": best_row,
                                                "score": best_score * 2
                                            })

                        if matches_found:
                            for idx, match in enumerate(all_match_data):
                                with st.expander(f"Match #{idx+1} (Confidence: {round(1 - (match['score']/2), 2)})"):
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.markdown("**📄 PDF Content:**")
                                        st.markdown(f"> {match['pdf_text']}")
                                    with col2:
                                        st.markdown("**🧬 Matched Content Details:**")
                                        match_data = match['db_match']
                                        
                                        def extract_focused_values(data):
                                            # Mapping of user requested keys to actual paths in the JSON
                                            key_mapping = {
                                                "input": ["input"],
                                                "classification": ["classification"],
                                                "pediatric code": [["classification", "pediatric_code"]],
                                                "rational summary": ["rationale_summary", "rational_summary"],
                                                "relavent mappings": ["relevant_mappings", "relavent_mappings"],
                                                "icd10pcs_codes": [["relevant_mappings", "icd10pcs_codes"]],
                                                "hcpcs_level2_codes": [["relevant_mappings", "hcpcs_level_2_codes"], ["relevant_mappings", "hcpcs_level2_codes"]],
                                                "why_relevant": [["relevant_mappings", "cpt_codes", 0, "why_relevant"], "why_relevant", "why_relavent"]
                                            }

                                            def get_nested(obj, path):
                                                curr = obj
                                                for p in path:
                                                    if isinstance(curr, dict) and p in curr:
                                                        curr = curr[p]
                                                    elif isinstance(curr, list) and isinstance(p, int) and len(curr) > p:
                                                        curr = curr[p]
                                                    else:
                                                        return None
                                                return curr

                                            def get_row_values(item):
                                                if not isinstance(item, dict): return None
                                                source = item
                                                if 'content' in item and isinstance(item['content'], dict):
                                                    source = item['content']
                                                    
                                                values = []
                                                for display_name, candidates in key_mapping.items():
                                                    val = None
                                                    for cand in candidates:
                                                        if isinstance(cand, list):
                                                            val = get_nested(source, cand)
                                                        else:
                                                            val = source.get(cand)
                                                        if val is not None: break
                                                    
                                                    if val is not None:
                                                        # Convert JSON to compact string in the table
                                                        if isinstance(val, (dict, list)):
                                                            s_val = json.dumps(val) # Compact string
                                                        else:
                                                            s_val = str(val)
                                                        values.append(s_val)
                                                return values if values else None

                                            if isinstance(data, dict):
                                                res = get_row_values(data)
                                                return [res] if res else []
                                            elif isinstance(data, list):
                                                rows = []
                                                for item in data:
                                                    res = get_row_values(item)
                                                    if res: rows.append(res)
                                                return rows
                                            return []

                                        import ast
                                        def robust_json_loads(val):
                                            if not isinstance(val, str): return val
                                            try:
                                                return json.loads(val)
                                            except json.JSONDecodeError:
                                                try:
                                                    return ast.literal_eval(val)
                                                except:
                                                    return val

                                        def extract_summary_table(data_obj):
                                            summary_data = []
                                            source = data_obj
                                            # If it's a dict from DB, it might have 'content'
                                            if isinstance(data_obj, dict) and 'content' in data_obj:
                                                source = robust_json_loads(data_obj['content'])
                                            
                                            if not isinstance(source, dict):
                                                return summary_data

                                            # ICD10CM Code
                                            icd = None
                                            if "input" in source and isinstance(source["input"], dict):
                                                icd = source["input"].get("icd10cm_code")
                                            if not icd:
                                                icd = source.get("icd10cm_code")
                                            
                                            if icd:
                                                summary_data.append({"Key": "icd10cm_code", "Value": icd})
                                            
                                            # CPT Codes
                                            cpt_list = []
                                            rel_mappings = source.get("relevant_mappings", {})
                                            if isinstance(rel_mappings, dict):
                                                mapped_cpts = rel_mappings.get("cpt_codes", [])
                                                if isinstance(mapped_cpts, list):
                                                    for c in mapped_cpts:
                                                        if isinstance(c, dict):
                                                            code = c.get("code")
                                                            if code: cpt_list.append(str(code))
                                                        else:
                                                            cpt_list.append(str(c))
                                            
                                            if not cpt_list and "cpt_codes" in source:
                                                raw_cpt = source.get("cpt_codes", [])
                                                if isinstance(raw_cpt, list):
                                                    for c in raw_cpt:
                                                        if isinstance(c, dict): cpt_list.append(str(c.get("code", "")))
                                                        else: cpt_list.append(str(c))
                                                else:
                                                    cpt_list.append(str(raw_cpt))
                                                    
                                            if cpt_list:
                                                summary_data.append({"Key": "cpt_codes", "Value": ", ".join(cpt_list)})
                                                
                                            # HCPCS Level 2 Codes
                                            hcpcs_list = []
                                            if isinstance(rel_mappings, dict):
                                                hcpcs_raw = rel_mappings.get("hcpcs_level2_codes") or rel_mappings.get("hcpcs_level_2_codes")
                                                if isinstance(hcpcs_raw, list):
                                                    for h in hcpcs_raw:
                                                        if isinstance(h, dict):
                                                            code = h.get("code") or h.get("hcpcs_code")
                                                            if code: hcpcs_list.append(str(code))
                                                        else:
                                                            hcpcs_list.append(str(h))
                                            
                                            if not hcpcs_list:
                                                h_raw = source.get("hcpcs_level2_codes") or source.get("hcpcs_level_2_codes")
                                                if isinstance(h_raw, list):
                                                    for h in h_raw:
                                                        if isinstance(h, dict): hcpcs_list.append(str(h.get("code", "")))
                                                        else: hcpcs_list.append(str(h))
                                                elif h_raw:
                                                    hcpcs_list.append(str(h_raw))
                                                    
                                            if hcpcs_list:
                                                summary_data.append({"Key": "hcpcs_level2_codes", "Value": ", ".join(hcpcs_list)})
                                                
                                            return summary_data

                                        # Existing matching details display
                                        if isinstance(match_data, dict) and 'content' in match_data:
                                            try:
                                                content_val = match_data['content']
                                                parsed_content = robust_json_loads(content_val)
                                                focused_values = extract_focused_values(parsed_content)
                                                
                                                if focused_values:
                                                    for row_values in focused_values:
                                                        st.table(pd.DataFrame(row_values, columns=["Matched Details"]))
                                                else:
                                                    st.info("No matching detailed data found.")
                                            except Exception as e:
                                                st.warning(f"Note: Could not parse content detail - {e}")
                                                st.write(match_data['content'])
                                        else:
                                            try:
                                                parsed_match = robust_json_loads(match_data)
                                                focused_values = extract_focused_values(parsed_match)
                                                if focused_values:
                                                    for row_values in focused_values:
                                                        st.table(pd.DataFrame(row_values, columns=["Matched Details"]))
                                                else:
                                                    if isinstance(parsed_match, (dict, list)):
                                                        st.json(parsed_match)
                                                    else:
                                                        st.info(str(parsed_match))
                                            except:
                                                st.info(str(match_data))

                                        # NEW: Short Summary Table for ICD and CPT
                                        st.markdown("**📌 Code Summary:**")
                                        summary_rows = extract_summary_table(match_data)
                                        if summary_rows:
                                            st.table(pd.DataFrame(summary_rows))
                                        else:
                                            st.info("No ICD/CPT codes found in summary.")
                        else:
                            st.info("No high-confidence matches found.")

        os.unlink(tmp_path)


