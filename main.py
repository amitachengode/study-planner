import logging
import streamlit as st
import os
from google import genai
from google.genai import types
from src.agent import Agent
from src.database import Database
logging.getLogger("transformers.models").setLevel(logging.ERROR)
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Syllabus RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 KTU Syllabus RAG Assistant")

@st.cache_resource
def initialize_systems():
    db = Database(name="ktu_syllabus", database_path="./chroma")
    api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("Please set the GEMINI_API_KEY environment variable or secret to proceed.")
        st.stop()
    return db, api_key

db_instance, api_key = initialize_systems()

if "my_agent" not in st.session_state:
    system_instruction = (
        "You are an expert academic assistant specializing in the KTU syllabus. "
        "Use the provided context chunks (retrieved via hybrid semantic and keyword search) "
        "to answer the user's questions accurately. If the context does not contain "
        "the answer, state that clearly."
        "if the context does not contain the answer, state that clearly."
        "do not make up topics or topics that are not in the context."
        "you may only use the information in the context to answer the question."
        "you can browse about the topics in the context."
    )
    # Start with gemini-2.5-flash as default, falling back automatically if needed
    st.session_state.my_agent = Agent(
        api_key=api_key,
        model_name="gemini-2.5-flash", 
        system_instruction=system_instruction
    )

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Hello! Ask me any question regarding your syllabus modules or topics."}
    ]

for message in st.session_state.ui_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your syllabus..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.ui_messages.append({"role": "user", "content": user_query})

    with st.spinner("Searching syllabus database..."):
        relevant_chunks = db_instance.search_hybrid(user_query, k=4)
        
    if relevant_chunks:
        context_str = "\n\n---\n\n".join(relevant_chunks)
        augmented_prompt = f"Context:\n{context_str}\n\nQuestion: {user_query}"
    else:
        augmented_prompt = user_query

    with st.chat_message("assistant"):
        with st.spinner("Formulating response..."):
            response_text = st.session_state.my_agent.ask(augmented_prompt, max_tokens=2000)
            st.markdown(response_text)
            
    st.session_state.ui_messages.append({"role": "assistant", "content": response_text})