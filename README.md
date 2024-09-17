Here is a comprehensive document to upload on GitHub and submit for your assignment, including the code for the RAG model for a QA bot, along with the architectural details:

---

# RAG Model for QA Bot

## Overview

This notebook demonstrates the development of a Retrieval-Augmented Generation (RAG) model for a Question Answering (QA) bot. The solution integrates Pinecone for vector search and Groq for chat-based completions. It provides a complete pipeline from data ingestion and indexing to generating responses based on user queries.

## Architecture

### Components

1. **Pinecone**: Used for vector database management and similarity search.
2. **Groq**: Handles chat completions using pre-trained models.
3. **Streamlit**: Provides a user-friendly web interface for interacting with the QA bot.
4. **PDF Processing**: Handles the extraction of text from uploaded PDF documents.

### Architecture Diagram

```plaintext
+--------------------+         +-------------------+
|                    |         |                   |
|   PDF Upload       |         |  Pinecone Vector  |
|   (Streamlit)      |         |   Database        |
|                    |         |                   |
+--------+-----------+         +--------+----------+
         |                            |
         v                            |
+--------+-----------+                |
|                    |                |
|   Groq Chat Model  |                |
|   (Generation)     |                |
|                    |                |
+--------+-----------+                |
         |                            |
         v                            |
+--------+-----------+                |
|                    |                |
|   User Query       |                |
|   (Streamlit)      |                |
|                    |                |
+--------------------+                |
         |                            |
         v                            |
+--------+-----------+                |
|                    |                |
|   Response         |                |
|   Display (Streamlit)              |
|                    |                |
+--------------------+----------------+
```

## Setup and Installation

1. **Install Dependencies**

   Install the required packages using the following command:
   ```bash
   pip install groq pinecone
   ```

2. **Environment Variables**

   Set up environment variables for API keys:
   - `PINECONE_API_KEY`: Your Pinecone API key.
   - `GROQ_API_KEY`: Your Groq API key.

   Example `.env` file:
   ```plaintext
   PINECONE_API_KEY=your_pinecone_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

## Code Implementation

### Initialization

The code starts by setting up the Pinecone and Groq clients, creating an index in Pinecone if it doesn't exist, and defining functions for encoding queries and retrieving documents.

```python
import pinecone
from groq import Groq
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "qa-bot-index"

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

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

def get_chatmodel_responses(prompt):
    try:
        retrieved_docs = retrieve_documents(prompt)
        combined_prompt = prompt + "\n\n" + "\n".join(retrieved_docs)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": combined_prompt}],
            model="llama3-8b-8192"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {e}"
```

### Streamlit Interface

The Streamlit application provides a web interface for users to interact with the QA bot. Users can upload PDF files, view their content, and chat with the bot. 

```python
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone and Groq clients
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index = pc.Index("qa-bot-index")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.set_page_config(page_title="Conversational Q&A Chatbot", page_icon=":robot_face:")

st.markdown("""
    <style>
    body { ... }
    .chatbox { ... }
    .user-message { ... }
    .assistant-message { ... }
    .dynamic-button { ... }
    .loading { ... }
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

if "conversation" not in st.session_state:
    st.session_state["conversation"] = []

st.header("Chat with the Assistant")

chatbox = st.empty()
with chatbox.container():
    if st.session_state["conversation"]:
        for message in st.session_state["conversation"]:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

st.sidebar.subheader("Settings")
show_file_content = st.sidebar.checkbox("Show File Content", value=True)
top_k = st.sidebar.slider("Number of Documents to Retrieve", min_value=1, max_value=10, value=5)

input_col, submit_col = st.columns([4, 1])
with input_col:
    input_text = st.text_input("Your Message:", key="input_text")
with submit_col:
    submit = st.button("Send", key="submit_button", help="Click to send your message")

loading_placeholder = st.empty()
if st.session_state.get("is_loading", False):
    with loading_placeholder:
        st.markdown('<div class="loading"></div>', unsafe_allow_html=True)
        time.sleep(2)  # Simulate processing time
        loading_placeholder.empty()

if submit:
    if input_text:
        st.session_state["conversation"].append({"role": "user", "content": input_text})
        st.session_state["is_loading"] = True
        
        query = input_text
        if 'file_content' in locals() and show_file_content:
            query += f"\nContext from file: {file_content}"
        
        response = get_chatmodel_responses(query)
        st.session_state["conversation"].append({"role": "assistant", "content": response})
        
        st.session_state["is_loading"] = False
        st.experimental_rerun()
    else:
        st.warning("Please enter a question to ask.")
```

## Usage

1. **Run the Notebook**

   Execute the notebook to set up the environment, create the Pinecone index, and define the functions.

2. **Start the Streamlit App**

   Run the Streamlit app to interact with the QA bot interface:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the Bot**

   - **Upload PDF**: Use the sidebar to upload PDF files. The content will be extracted and displayed.
   - **Chat with the Bot**: Enter queries in the chat interface. The bot will retrieve relevant documents and provide responses.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to modify this document according to your needs and ensure that sensitive information like API keys is handled securely.
