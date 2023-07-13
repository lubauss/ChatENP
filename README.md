# ChatENP: An Augmented Search Assistant

ChatENP is a conversational AI application powered by Streamlit, OpenAI, and Pinecone. It uses GPT-3.5-turbo-16k for generating responses and a retrieval-based QA model to search a given knowledge base. The knowledge base is a set of documents uploaded by the user, and the AI uses these documents to answer queries.

## Getting Started

### Prerequisites
- Python 3.7 or newer.

### Installation

1. **Clone this repository to your local machine:**
    ```
    git clone https://github.com/yourusername/ChatENP.git
    cd ChatENP
    ```

2. **Create a virtual environment:**
    ```
    python3 -m venv ChatENP
    source ChatENP/bin/activate  # On Windows, use `ChatENP\Scripts\activate`
    ```

3. **Install the required packages:**
    ```
    pip install -r requirements.txt
    ```

4. **Set up secrets file:**

    Create a `.streamlit` directory at the root of your project. Inside the `.streamlit` directory, create a `secrets.toml` file and add your API keys:
    ```toml
    # secrets.toml
    [secrets]
    OPENAI_API_KEY = "your-openai-api-key"
    SERPAPI_API_KEY = "your-serpapi-api-key"
    PINECONE_API_KEY = "your-pinecone-api-key"
    PINECONE_API_ENV = "your-pinecone-api-env"
    ```
    Please replace `"your-openai-api-key"`, `"your-serpapi-api-key"`, `"your-pinecone-api-key"`, and `"your-pinecone-api-env"` with your actual API keys.

5. **Run the Streamlit app:**
    ```
    streamlit run ChatENP.py
    ```
6. Navigate to the provided local URL in your web browser.

### Using the App

- **Upload your documents:** In the sidebar, you can upload your PDF documents that will be used as the knowledge base for answering queries. Enter a Pinecone namespace and click on "Process". The app will then extract the text from the uploaded PDFs, create embeddings for the text, and store these embeddings in a Pinecone index.
  
- **Choose a namespace to query:** In the sidebar, you can select the namespace from which you want to retrieve information. The namespaces correspond to the different document sets you've uploaded.
  
- **Ask a question:** In the main panel, enter your question in the text area labeled "Query" and click "Send". The app will use the GPT-3.5-turbo-16k model to generate a response, querying the document set when necessary.
