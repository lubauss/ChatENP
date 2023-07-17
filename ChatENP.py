import streamlit as st
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from tqdm.auto import tqdm
from uuid import uuid4
from time import sleep
import os
import hashlib
from streamlit_chat import message

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__586518e17ecd47e4a44b8767a151d26a"

# Global variables
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
            environment=os.environ['PINECONE_API_ENV'])

index_name = 'langchain-retrieval-agent'
index = pinecone.Index(index_name)
embed = OpenAIEmbeddings(model='text-embedding-ada-002')

tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# Set the chunk size and overlap
chunk_size = 700
chunk_overlap = 100

# Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

def get_tokens(text:str):
  embedding_encoding="cl100k_base"
  encoding=tiktoken.get_encoding(embedding_encoding)
  return(len(encoding.encode(text)))

def get_pdf_pages(pdf_docs):
    data=[]

    for pdf in pdf_docs:
        # print the original file name
        original_filename = os.path.splitext(pdf.name)[0]
        st.write(f'Processing file: {original_filename}')

        # sanitize the file name
        safe_filename = "".join(c if c.isalnum() else "_" for c in original_filename)

        # create a temporary file ended with '.pdf'
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', prefix=safe_filename + '_') as fp:
            fp.write(pdf.getvalue())
            # use the temporary file's name instead of pdf.name
            docname = fp.name
        loader = PyPDFLoader(docname)
        pages = loader.load()

        for page in pages:
            page.metadata['page'] += 1  # increment page number

        m = hashlib.md5()  # this will convert doc chunk number into unique ID

        for page in tqdm(pages):
            chunk_source = f"{original_filename}.pdf, page: {page.metadata['page']}"
            m.update(chunk_source.encode('utf-8'))
            uid = m.hexdigest()[:12]
            chunks = text_splitter.split_text(page.page_content)
            for i, chunk in enumerate(chunks):
                data.append({
                    'id': f'{uid}-{i}',
                    'text': chunk,
                    'source': chunk_source
                })

    return data

def pinecone_index(pinecone_namespace, data, index_name):

    # Initialize the index here
    pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
            environment=os.environ['PINECONE_API_ENV'])
    index = pinecone.Index(index_name)

    batch_limit = 50  # how many embeddings we create and insert at once

    texts = []
    metadatas = []

    # Inicia el bucle para cargar los embeddings de texto a Pinecone
    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'uid': str(record['id']),
            'source': record['source']
        }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'])
        # create individual metadata dicts for each chunk
        record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index = index
            index.upsert(vectors=zip(ids, embeds, metadatas), namespace=pinecone_namespace)
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas), namespace=pinecone_namespace)

def get_vectorstore(query_namespace, index):
    embed = OpenAIEmbeddings(model='text-embedding-ada-002')
    text_field = "text"
    vectorstore = Pinecone(index, embed.embed_query, text_field, namespace=query_namespace)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm_1 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    # Retrieval qa chain
    qa = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_1,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    return qa

def run_qa(qa, query):
    result = qa({"question":query})
    return f"{result['answer']}\n Sources:\n {result['sources']}"

def get_agent(run_qa_func):
    llm_2 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    
    # Calculator
    #llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    tools = [
        Tool(
            name='Knowledge Base',
            func=run_qa_func,
            description="use this tool when answering general knowledge queries to get more information about the topic",
            return_direct=True,
        ),

    ]
   
    # Conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=7,
        return_messages=True
    )
    agent = initialize_agent(tools, llm_2, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, max_iterations=3,early_stopping_method='generate', memory=conversational_memory,handle_parsing_errors=True)

    return agent

def handle_chat(agent, query):
    response = agent.run(input=query)
    st.session_state["agent"] = agent
    st.session_state.requests.append(query)
    st.session_state.responses.append(response)
    

def main():
    global index, index_name

    st.header("💻 ChatENP 🍺 \n ✨📚 **Asistente de Búsqueda Aumentada** 📖✨")

    query_namespace = None  # Initialize query_namespace to None

    with st.sidebar:
        st.subheader("⬆️ Sube tus documentos 📄")
        pdf_docs = st.file_uploader(
            "📄 Arrastra aquí PDFs y haz clic 👆 en 'Procesar':", accept_multiple_files=True)
        upload_namespace = st.text_input("📝 Escribe un nombre para tu base de datos 🗄️:")

        if st.button("Procesar ⚙️"):
            if not upload_namespace:
                st.warning("🎗️ No olvides  el nombre de tu base de datos 🗄️")
            else:
                with st.spinner("Procesando... ⏳"):
                    # get pdf text
                    docs_pages = get_pdf_pages(pdf_docs)
                    st.write(docs_pages)

                    # create vector store
                    pinecone_index(upload_namespace, docs_pages, index_name)
        # Always show the header
        st.subheader("Haz clic 👆 en la base de datos 🗄️ que quieras consultar 🔍")

        pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
                environment=os.environ['PINECONE_API_ENV'])

        # Obtain the existing namespaces
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {}).keys()

        # Initialize st.session_state['namespace'] if it doesn't exist
        if 'namespace' not in st.session_state:
            st.session_state['namespace'] = None

        # Create a button for each namespace
        for namespace in namespaces:
            if st.button(namespace):
                # Store the current selection in session state
                st.session_state['namespace'] = namespace
                # Update query_namespace
                query_namespace = st.session_state['namespace']
        # Initialize the agent only if it doesn't exist in the session state or the namespace has changed
        if 'agent' not in st.session_state or st.session_state['namespace'] != query_namespace:
            # Initialize your agent here
            query_namespace = st.session_state['namespace']
            vectorstore = get_vectorstore(query_namespace, index)        
            qa = get_conversation_chain(vectorstore)
            agent = get_agent(lambda query: run_qa(qa, query)) # Pass run_qa as a lambda function
            st.session_state['agent'] = agent # Set the agent here
        else:
            agent = st.session_state['agent'] # If the agent already exists in the session state, get it.


        # Check if the namespace has been selected
        if 'namespace' in st.session_state and st.session_state['namespace']:
            st.success(f"Base de datos 🗄️ seleccionada ✅: {st.session_state['namespace']}")
        else:
            st.warning("No olvides 🎗️ seleccionar 👆 la base de datos 🗄️ que quieres consultar.")
            st.stop()  # Stop execution of the script

    if 'responses' not in st.session_state:
        st.session_state['responses'] = []
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # container for text box
    response_container = st.container()
    textcontainer = st.container()

    with response_container:
        if st.session_state['responses']:        
            for i in range(len(st.session_state['requests'])):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
                if i < len(st.session_state['responses']):
                    message(st.session_state['responses'][i], key=str(i))
        
    with textcontainer:
    
        query = st.text_area("Consulta 🔍: ", key="input")
        send_button = st.button("Enviar 📤")

        if send_button and query:
            with st.spinner("Buscando... 🔎"):
                handle_chat(agent, query)

            # Add a small delay
            sleep(0.1)

            # Rerun the script
            st.experimental_rerun()

if __name__ == '__main__':
    main() 
