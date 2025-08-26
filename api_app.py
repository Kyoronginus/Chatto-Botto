import streamlit as st
import os
# Import both ChatHuggingFace and HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Chatto-Botto")

# Get API token from secrets or environment variables
def get_huggingface_token():
    try:
        # Try to get from Streamlit secrets first (for local development)
        return st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    except:
        # Fall back to environment variable (for Hugging Face Spaces)
        token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            st.error("❌ Hugging Face API token not found! Please set it in your Hugging Face Space secrets.")
            st.stop()
        return token

# Secrets setup instructions
# For local development: Create a file named .streamlit/secrets.toml and add: HUGGINGFACEHUB_API_TOKEN = "hf_your_token_here"
# For Hugging Face Spaces: Add HUGGINGFACEHUB_API_TOKEN as a secret in your Space settings

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar
with st.sidebar:
    st.header("Model Selection")
    selected_model = st.selectbox(
        "Choose an LLM:",
        ["google/gemma-2-9b-it", "mistralai/Mistral-7B-Instruct-v0.2"]
    )

    if st.button("Summarize Chat"):
        if st.session_state.messages:
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            summarization_prompt = f"Please provide a concise summary of the following conversation:\n---\n{chat_history}\n---\nSummary:"
            
            # CORRECT INITIALIZATION FOR SUMMARIZER
            summarizer_endpoint = HuggingFaceEndpoint(
                repo_id=selected_model,
                huggingface_api_token=get_huggingface_token(),
                temperature=0.5,
                max_new_tokens=150
            )
            llm_summarizer = ChatHuggingFace(llm=summarizer_endpoint)
            
            summary = llm_summarizer.invoke(summarization_prompt)
            st.subheader("Chat Summary")
            st.write(summary.content)
        else:
            st.warning("No chat history to summarize.")

    with st.expander("⚙️ Advanced Settings"):
        temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.7, step=0.05)
        
        # Add these lines for top_p and top_k
        top_p = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=50, step=1)
        
        max_tokens = st.number_input("Max Tokens", min_value=64, value=512, key="max_tokens_main")

# React to user input
if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # CORRECT INITIALIZATION FOR MAIN CHAT
    # Step 1: Create the endpoint object with all parameters
    llm_endpoint = HuggingFaceEndpoint(
        repo_id=selected_model,
        huggingface_api_token=get_huggingface_token(),
        temperature=temperature,
        max_new_tokens=max_tokens
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    response = llm.invoke(prompt)
    
    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})