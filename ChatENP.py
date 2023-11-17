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

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__586518e17ecd47e4a44b8767a151d26a"
os.environ["LANGCHAIN_SESSION"] = "chat-enp-streamlit"

# Global variables
pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
            environment=os.environ['PINECONE_API_ENV'])

index_name = 'langchain-retrieval-agent-index'
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
chunk_size = 500
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
    vectorstore = Pinecone(
        namespace=query_namespace,
        index=index, 
        embedding=embed,
        text_key=text_field,
)
    
    return vectorstore

def get_conversation_chain(vectorstore):
    llm_1 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    # Retrieval qa chain New Template
    template ="""
    From document snippets and a query, yield a sourced answer. If unsure, state it; always cite 'SOURCES'.
    QUESTION: What law governs the interpretation of this contract?
    =========
    Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in relation to any dispute (contractual or non-contractual) concerning this Agreement. Either party may apply to any court for an injunction or other relief to protect its Intellectual Property Rights.
    Source: 28-pl
    Content: No Waiver. Any delay in exercising any right under this Agreement is not a waiver. Severability clause present. No Agency or Third-Party Beneficiaries clauses included.
    Source: 30-pl
    Content: Google may believe, in good faith, the Distributor violated or could likely violate Anti-Bribery Laws as defined in Clause 8.5.
    Source: 4-pl
    =========
    FINAL ANSWER: This Agreement is governed by English law.
    SOURCES: 28-pl

    QUESTION: What did the president say about Michael Jackson?
    =========
    Content: Madam Speaker, Madam Vice President, First Lady, Second Gentleman, Congress, Cabinet, Justices, fellow Americans. Last year COVID-19 separated us, this year we're united. Meeting as Democrats, Republicans, Independents, but chiefly as Americans. Duty to each other, the public, and the Constitution. Freedom will triumph over tyranny. Six days ago, Putin miscalculated Ukraine's resolve.
    Source: 0-pl
    Content: Losses to COVID-19, both time and lives. Moment for a reset; not a partisan issue. NYPD visit after Officer Mora and Rivera's funerals.
    Source: 24-pl
    Content: Russian invasion has global implications. Sanctions targeted at Russia's economy. Released 60M barrels of oil in coordination with allies.
    Source: 5-pl
    Content: Advocacy for patient support. Calls for ARPA-H funding for health breakthroughs. Unity agenda; we're gathered in the citadel of our democracy.
    Source: 34-pl
    =========
    FINAL ANSWER: The president did not mention Michael Jackson.
    SOURCES:

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:
    """

    qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm_1,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                template=template,
                input_variables=["summaries", "question"],
            ),
        },
    )

    return qa

def handle_chat(qa, query):
    result = qa({"question": query})
    response = f"{result['answer']}  \nSources: {result['sources']}"
    
    # Store the qa function for future use (if needed)
    st.session_state["qa"] = qa

    return response  # Return the assistant's response for further use
    
def main():
    global index, index_name

    st.header("💻 ChatENP 🍺 \n ✨📚 **Asistente de Búsqueda Aumentada** 📖✨")

    query_namespace = None  # Initialize query_namespace to None

    with st.sidebar:
        st.subheader("⬆️ Sube tus documentos 📄")
        pdf_docs = st.file_uploader(
            "📄 Arrastre aquí sus PDFs y haga clic 👆 en 'Procesar':", accept_multiple_files=True)
        upload_namespace = st.text_input("📝 Escriba el nombre de su base de datos 🗄️:")

        if st.button("Procesar ⚙️"):
            if not upload_namespace:
                st.warning("🎗️ No olvide poner un nombre de su base de datos 🗄️")
            else:
                with st.spinner("Procesando... ⏳"):
                    # get pdf text
                    docs_pages = get_pdf_pages(pdf_docs)
                    st.write(docs_pages)

                    # create vector store
                    pinecone_index(upload_namespace, docs_pages, index_name)
        # Always show the header
        st.subheader("👆 Haga clic  en la base de datos 🗄️ que quiera consultar 🔍")

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

        # Initialize the qa_chain only if it doesn't exist in the session state or the namespace has changed
        if 'qa_chain' not in st.session_state or st.session_state['namespace'] != query_namespace:
            # Initialize your qa_chain here
            query_namespace = st.session_state['namespace']
            vectorstore = get_vectorstore(query_namespace, index)        
            qa_chain = get_conversation_chain(vectorstore)

            st.session_state['qa_chain'] = qa_chain # Set the qa_chain here
        else:
            qa_chain = st.session_state['qa_chain'] # If the qa_chain already exists in the session state, get it.


        # Check if the namespace has been selected
        if 'namespace' in st.session_state and st.session_state['namespace']:
            st.success(f"Base de datos 🗄️ seleccionada ✅: {st.session_state['namespace']}")
        else:
            st.warning("🎗️ No olvide seleccionar la base de datos 👆 a consultar 🗄️.")
            st.stop()  # Stop execution of the script

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages using `st.chat_message`
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Cual es su consulta?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = handle_chat(qa_chain, prompt)  # Assuming handle_chat returns the full assistant's response
            
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == '__main__':
    main() 
