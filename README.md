
---

# RAG Model for QA Bot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) model for a Question Answering (QA) chatbot. It leverages Pinecone for vector storage and retrieval, Groq for generating responses, and Streamlit for creating an interactive user interface. The architecture integrates various technologies to provide a seamless conversational experience.

## Architecture

### Components

1. **Pinecone**: Used for vector storage and retrieval. It stores document vectors and allows efficient querying to retrieve relevant documents based on user queries.
2. **Groq**: Provides a conversational AI model for generating responses based on the retrieved documents and the user's query.
3. **Streamlit**: Powers the web interface where users can interact with the QA bot. It provides functionalities for uploading PDFs, querying the bot, and displaying responses.

### Architecture Diagram

```
+------------------+       +------------------+       +------------------+
|   User Interface | <---> | Retrieval System | <---> | Generation Model |
|   (Streamlit)    |       |    (Pinecone)    |       |     (Groq)       |
+------------------+       +------------------+       +------------------+
                           |                          |
                           +--> [Document Vectors] <--+
                           +--> [Query Vectors] ----> [Responses]
```

## Setup Instructions

### Prerequisites

1. **Python 3.10+**: Ensure you have Python 3.10 or newer installed.
2. **API Keys**: Obtain API keys for Pinecone and Groq.
3. **Environment Variables**: Store your API keys in environment variables for secure access.

### Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**

   Create a `requirements.txt` file with the following content:

   ```
   groq
   pinecone
   streamlit
   numpy
   pandas
   PyMuPDF
   python-dotenv
   ```

   Install the dependencies using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**

   Create a `.env` file in the root directory with the following content:

   ```ini
   PINECONE_API_KEY=<your-pinecone-api-key>
   GROQ_API_KEY=<your-groq-api-key>
   ```

### Usage

1. **Run the Streamlit Application**

   ```bash
   streamlit run app.py
   ```

2. **Interacting with the QA Bot**

   - **Upload PDF**: Use the sidebar to upload a PDF file. The content will be extracted and displayed.
   - **Query the Bot**: Enter your question in the text input field and click "Send" to get a response from the bot.
   - **Settings**: Adjust the number of documents to retrieve and view file content options in the sidebar.

## Code Overview

### 1. Dependencies and Initialization

```python
import pinecone
from groq import Groq
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import time
import os
from PIL import Image
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "qa-bot-index"

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
```

### 2. Pinecone Index Management

```python
# Create the index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="euclidean",
        spec=pinecone.ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
```

### 3. Vector Encoding and Retrieval

```python
def encode_query_to_vector(query):
    return np.random.rand(1536).tolist()  # Replace with actual encoding logic

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
```

### 4. Generating Responses

```python
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
```

### 5. Streamlit Interface

```python
st.set_page_config(page_title="Conversational Q&A Chatbot", page_icon=":robot_face:")

# Custom CSS for styling
st.markdown("""
    <style>
    /* CSS styling here */
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Chatbot Options")
st.sidebar.subheader("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# Handle PDF upload and display content
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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

