import streamlit as st
import os
import pinecone
import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq
import numpy as np
import pandas as pd
import time
from PIL import Image
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "qa-bot-index"

# Create the index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Ensure this matches your embedding dimension
        metric="euclidean",
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to encode queries to vectors (Placeholder implementation)
def encode_query_to_vector(query):
    return np.random.rand(1536).tolist()  # Replace with actual encoding logic

# Function to retrieve relevant documents from Pinecone
def retrieve_documents(query, top_k=5):
    query_vector = encode_query_to_vector(query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_values=True,
        include_metadata=True
    )
    retrieved_docs = [match['metadata']['text'] for match in results['matches']]
    return retrieved_docs

# Function to get responses from Groq API
def get_chatmodel_responses(prompt):
    try:
        retrieved_docs = retrieve_documents(prompt)
        combined_prompt = prompt + "\n\n" + "\n".join(retrieved_docs)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": combined_prompt}],
            model="llama3-8b-8192"
        )
        answer = chat_completion.choices[0].message.content
        return answer
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot", page_icon=":robot_face:")

# Add custom CSS for dynamic styling
st.markdown("""
    <style>
    body {
        background: #F5F5F5;
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chatbox {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 15px;
        background-color: #FFFFFF;
        color: #333;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user-message, .assistant-message {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        animation: fadeIn 0.5s ease-out;
    }
    .user-message {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .assistant-message {
        background-color: #FFFFFF;
        color: #333;
        border: 1px solid #E0E0E0;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .dynamic-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 8px 4px;
        cursor: pointer;
        border-radius: 6px;
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .dynamic-button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .dynamic-button:active {
        background-color: #388e3c;
        transform: scale(0.98);
    }
    .loading {
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 4px solid #4CAF50;
        border-top-color: transparent;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Chatbot Options")
st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None:
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    file_content = ""
    for page in pdf_document:
        file_content += page.get_text()
    pdf_document.close()
    st.sidebar.subheader("File Content")
    st.sidebar.write(file_content)

# Display conversation history
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

st.header("Chat with the Assistant")

# Display chat messages with animations
chatbox = st.empty()
with chatbox.container():
    if st.session_state["conversation"]:
        for message in st.session_state["conversation"]:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

# Add dynamic elements
st.sidebar.subheader("Settings")
show_file_content = st.sidebar.checkbox("Show File Content", value=True)
top_k = st.sidebar.slider("Number of Documents to Retrieve", min_value=1, max_value=10, value=5)

# Input field for manual questions with dynamic button
input_col, submit_col = st.columns([4, 1])
with input_col:
    input_text = st.text_input("Your Message:", key="input_text")
with submit_col:
    submit = st.button("Send", key="submit_button", help="Click to send your message")

# Real-time progress indicator
loading_placeholder = st.empty()
if st.session_state.get("is_loading", False):
    with loading_placeholder:
        st.markdown('<div class="loading"></div>', unsafe_allow_html=True)
        time.sleep(2)  # Simulate processing time
        loading_placeholder.empty()

# If the submit button is clicked
if submit:
    if input_text:
        st.session_state["conversation"].append({"role": "user", "content": input_text})
        st.session_state["is_loading"] = True
        
        # Combine user input with file content if available
        query = input_text
        if 'file_content' in locals() and show_file_content:
            query += f"\nContext from file: {file_content}"
        
        # Get the response from Groq API
        response = get_chatmodel_responses(query)
        
        # Append AI's response to the conversation history
        st.session_state["conversation"].append({"role": "assistant", "content": response})
        
        # Clear input field and update chat
        st.session_state["is_loading"] = False
        st.experimental_rerun()
    else:
        st.warning("Please enter a question to ask.")
