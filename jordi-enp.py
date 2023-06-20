import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())# read local .env

from langchain.vectorstores import Pinecone
import pinecone
from langchain.embeddings import OpenAIEmbeddings

embed = OpenAIEmbeddings(model='text-embedding-ada-002')

index_name = 'langchain-retrieval-agent'

pinecone.init(api_key=os.environ['PINECONE_API_KEY'], 
              environment=os.environ['PINECONE_API_ENV'])  

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

from langchain import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

OpenAI.openai_api_key = os.environ['OPENAI_API_KEY']
SERPAPI_API_KEY=os.environ['SERPAPI_API_KEY']

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Calculator
#llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# Retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description="use this tool when answering general knowledge queries to get more information about the topic",
    ),

]

from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=10,
    return_messages=True
)

agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, max_iterations=3,early_stopping_method='generate', memory=conversational_memory)


# Streamlit framework
import streamlit as st
from streamlit_chat import message

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize the agent only if it doesn't exist in the session state
if 'agent' not in st.session_state:
    # Initialize your agent here
    st.session_state['agent'] = agent
else:
    agent = st.session_state['agent']

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_area("Query: ", key="input")
    send_button = st.button("Send")

    if send_button and query:
        with st.spinner("typing..."):
            # Call the agent with the query
            response = agent.run(query)

            # The agent should already return 'chat_history' in its response, 
            # so you don't need to add it manually

            # Save the agent back to the session_state after the interaction
            st.session_state["agent"] = agent

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:        
        for i in range(len(st.session_state['requests'])):
            message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
            if i < len(st.session_state['responses']):
                message(st.session_state['responses'][i], key=str(i))

    