import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("Local AI Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


with st.sidebar:
        st.header("Model Selection")
        # Dropdown to choose the model
        selected_model = st.selectbox(
            "Choose an LLM:",
            ["tinyllama:1.1b", "smollm:1.7b"] 
        )


# Place this in the sidebar, below the model selection
if st.sidebar.button("Summarize Chat"):
    if st.session_state.messages:
        # Get the current chat history
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        
        # Create the summarization prompt
        summarization_prompt = f"""
        Please provide a concise summary of the following conversation:
        ---
        {chat_history}
        ---
        Summary:
        """
        
        # Use the currently selected model for summarization
        llm_summarizer = ChatOllama(model=selected_model)
        summary = llm_summarizer.invoke(summarization_prompt)
        
        # Display the summary
        st.sidebar.subheader("Chat Summary")
        st.sidebar.write(summary.content)
    else:
        st.sidebar.warning("No chat history to summarize.")

with st.sidebar.expander("⚙️ Advanced Settings"):
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    top_p = st.slider("Top-P", min_value=0.1, max_value=1.0, value=0.9, step=0.05)
    top_k = st.slider("Top-K", min_value=1, max_value=100, value=40, step=1)
    max_tokens = st.number_input("Max Tokens", min_value=64, value=2048)


# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    llm = ChatOllama(
        model=selected_model,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    
    
    response = llm.invoke(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})