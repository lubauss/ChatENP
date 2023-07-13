import streamlit as st
from PyPDF2 import PdfReader
import tiktoken
from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from tqdm.auto import tqdm
import uuid
from time import sleep
import os
from streamlit_chat import message

from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Global variables
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
            environment=os.environ['PINECONE_API_ENV'])
index_name = 'langchain-retrieval-agent'
index = pinecone.Index(index_name)

def get_tokens(text:str):
  embedding_encoding="cl100k_base"
  encoding=tiktoken.get_encoding(embedding_encoding)
  return(len(encoding.encode(text)))

def get_pdf_pages(pdf_docs):
    text = ""
    data=[]
    for pdf in pdf_docs:
        docname= pdf.name
        reader = PdfReader(pdf)
        count_page=1
        for page in reader.pages:
            text=page.extract_text()
            if not text=="":
            # text += page.extract_text() + "\n"
                tokens=get_tokens(text)
                data.append({"document":docname,"page":count_page,"text":text,"tokens":tokens})
                count_page+=1
    return data

def pinecone_index(pdf_docs, pinecone_namespace, data, index_name, index):
    import openai

    embed = OpenAIEmbeddings(model='text-embedding-ada-002')

    if index_name not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
            name=index_name,
            metric='cosine',
            dimension=1536,  # 1536 dim of text-embedding-ada-002
            metadata_config={'indexed': ['document', 'page']}
        )
    
    index.describe_index_stats()

    embed_model = "text-embedding-ada-002"

    batch_size = 1  # how many embeddings we create and insert at once

    # Inicia el bucle para cargar los embeddings de texto a Pinecone
    with pinecone.Index(index_name= index_name) as index:
        for i in tqdm(range(0, len(data), batch_size)):
            i_end = min(len(data), i+batch_size)
            meta_batch = data[i:i_end]

            # Obtener los IDs de cada documento
            ids_batch = [str(uuid.uuid4()) for x in meta_batch]
            
            # Obtener los textos a codificar
            texts = [x["text"] for x in meta_batch]

            # Crear embeddings con OpenAI
            done = False
            while not done:
                try:
                    res = openai.Embedding.create(input=texts, engine=embed_model)
                    done = True
                except Exception as e:
                    print(f"Error during embedding creation: {e}")
                    sleep(5.0)
            '''
                except Exception as e:
                print(e)
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except Exception as e:
                        print(e)
            '''
            embeds = [record['embedding'] for record in res['data']] 

            # Crear una lista de tuplas con los IDs, los embeddings y los metadatos
            to_upsert = list(zip(ids_batch, embeds, meta_batch))

            # Upsert to Pinecone
            index.upsert(vectors=to_upsert, namespace=pinecone_namespace)

    # After the indexing operation:
    if os.path.exists('namespace_docs.txt'):
        with open('namespace_docs.txt', 'r') as file:
            namespace_docs = dict(line.strip().split(':', 1) for line in file)
    else:
        namespace_docs = {}

    new_docs = [doc.name for doc in pdf_docs]
    existing_docs = namespace_docs.get(pinecone_namespace, "").split(',')

    namespace_docs[pinecone_namespace] = ",".join(list(set(existing_docs + new_docs)))

    with open('namespace_docs.txt', 'w') as file:
        for namespace, docs in namespace_docs.items():
            file.write(f"{namespace}:{docs}\n")

def get_vectorstore(query_namespace, index):
    embed = OpenAIEmbeddings(model='text-embedding-ada-002')
    text_field = "text"
    # switch back to normal index for langchain
    index = pinecone.Index(index_name)
    vectorstore = Pinecone(index, embed.embed_query, text_field, namespace=query_namespace)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm_1 = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k")

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
    llm_2 = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo-16k")
    
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
    global index
    st.header("💻 ChatENP 🍺 \n 📚 **Asistente de Búsqueda Aumentada** 📖")

    st.write(
        "Environment variables:",
        os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"],
        os.environ["SERPAPI_API_KEY"] == st.secrets["SERPAPI_API_KEY"],
        os.environ["PINECONE_API_KEY"] == st.secrets["PINECONE_API_KEY"],
        os.environ["PINECONE_API_ENV"] == st.secrets["PINECONE_API_ENV"],
    )

    with st.sidebar:
        st.subheader("Sube tus documentos")
        pdf_docs = st.file_uploader(
            "Arrastra aquí PDFs y dale click a 'Procesar'", accept_multiple_files=True)
        upload_namespace = st.text_input("Escribe un nombre para tu base de datos:")

        if st.button("Procesar"):
            if not upload_namespace:
                st.warning("No olvides el nombre de tu base de datos")
            else:
                with st.spinner("Procesando"):
                    # get pdf text
                    docs_pages = get_pdf_pages(pdf_docs)
                    st.write(docs_pages)

                    # create vector store
                    vector_namespace = pinecone_index(pdf_docs, upload_namespace, docs_pages, index)
        # Always show the header
        st.subheader("Selecciona una base de datos que quieras consultar")

        index_name = 'langchain-retrieval-agent'

        pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
                environment=os.environ['PINECONE_API_ENV'])

        # Obtain the existing namespaces
        index = pinecone.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get('namespaces', {}).keys()

        # Create a button for each namespace
        for namespace in namespaces:
            if st.button(namespace):
                # Store the current selection in session state
                st.session_state['namespace'] = namespace

        # Check if the namespace has been selected
        if 'namespace' in st.session_state and st.session_state['namespace']:
            st.success(f"Base de datos seleccionada: {st.session_state['namespace']}")
        else:
            st.warning("No olvides seleccionar la base de datos que quieres consultar.")
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
    # Initialize the agent only if it doesn't exist in the session state
        if 'agent' not in st.session_state:
            # Initialize your agent here
            query_namespace = st.session_state['namespace']
            vectorstore = get_vectorstore(query_namespace, index)        
            qa = get_conversation_chain(vectorstore)
            agent = get_agent(lambda query: run_qa(qa, query)) # Pass run_qa as a lambda function
            st.session_state['agent'] = agent # Set the agent here
        else:
            agent = st.session_state['agent'] # If the agent already exists in the session state, get it.

        query = st.text_area("Consulta: ", key="input")
        send_button = st.button("Enviar")

        if send_button and query:
            with st.spinner("Buscando..."):
                handle_chat(agent, query)

            # Add a small delay
            sleep(0.1)

            # Rerun the script
            st.experimental_rerun()

if __name__ == '__main__':
    main() 
