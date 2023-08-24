import os
import streamlit as st
import uuid

from time import time

from config import AppConfig

from redisvl.llmcache.semantic import SemanticCache
from redisvl.vectorize.text import VertexAITextVectorizer

from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.embeddings import VertexAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from dotenv import load_dotenv
load_dotenv()


# Load Global env

load_dotenv()

config = AppConfig()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex


# Helpers

@st.cache_resource()
def configure_retriever(path):
    """Create the Redis Vector DB retrieval tool"""
    # Read documents
    docs = []
    for file in os.listdir(path):
        print(file, flush=True)
        loader = PyPDFLoader(os.path.join(path, file))
        docs.extend(loader.load())
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    # Create embeddings and store in vectordb
    embeddings = VertexAIEmbeddings(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
    vectordb = Redis.from_documents(
        splits, embeddings, redis_url=config.REDIS_URL, index_name="chatbot"
    )
    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": config.RETRIEVE_TOP_K})
    tool = create_retriever_tool(retriever, "search_chevy_manual", "Searches and returns snippets from the Chevy Colorado 2022 car manual.")
    return tool


@st.cache_resource()
def configure_cache():
    """Set up the Redis LLMCache built with VertexAI Text Embeddings"""
    llmcache_embeddings = VertexAITextVectorizer(
        api_config={"project_id": config.GCP_PROJECT_ID, "location": config.GCP_LOCATION}
    )
    return SemanticCache(
        redis_url=config.REDIS_URL,
        threshold=config.LLMCACHE_THRESHOLD, # semantic similarity threshold
        vectorizer=llmcache_embeddings
    )


def configure_agent(chat_memory, tools: list):
    """Configure the conversational chat agent that can use the Redis vector db for RAG"""
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=chat_memory, return_messages=True
    )
    chatLLM = ChatVertexAI(
        temperature=0.1,
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION
    )
    PREFIX = """"You are a friendly AI assistant that can help you understand your Chevy 2022 Colorado vehicle based on the provided PDF car manual. Users can ask questions of your manual! You should not make anything up."""

    FORMAT_INSTRUCTIONS = """You have access to the following tools:

    {tools}

    Use the following format:

    '''
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    '''

    When you have gathered all the information required, respond to the user in a friendly manner.
    """

    SUFFIX = """

    Begin! Remember to give detailed, informative answers

    Previous conversation history:
    {chat_history}

    New question: {input}
    {agent_scratchpad}
    """
    return initialize_agent(
        tools,
        chatLLM,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            'prefix': PREFIX,
            'format_instructions': FORMAT_INSTRUCTIONS,
            'suffix': SUFFIX
        }
    )


class PrintRetrievalHandler(BaseCallbackHandler):
    """Callback to print retrieved source documents from Redis during RAG."""
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)


def generate_response(
    use_cache: bool,
    llmcache: SemanticCache,
    user_query: str,
    agent
) -> str:
    """Generate a response to the user's question after checking the cache (if enabled)."""
    t0 = time()
    if use_cache:
        if response := llmcache.check(user_query):
            print("Cache Response Time (secs)", time()-t0, flush=True)
            return response[0]

    retrieval_handler = PrintRetrievalHandler(st.container())
    response = agent.run(input=user_query, callbacks=[retrieval_handler])
    print("Full Response Time (secs)", time()-t0, flush=True)
    return response


def render():
    """Render the Streamlit chatbot user interface."""
    # Main Page
    st.set_page_config(page_title=config.PAGE_TITLE, page_icon=config.PAGE_ICON)
    st.title(config.PAGE_TITLE)

    # Setup LLMCache in Redis
    llmcache = configure_cache()

    # Setup Redis memory for conversation history
    msgs = RedisChatMessageHistory(
        session_id=st.session_state.session_id, url=config.REDIS_URL
    )

    # Sidebar
    with st.sidebar:
        use_cache = st.checkbox("Use LLM cache?")
        if st.button("Clear LLM cache"):
            llmcache.clear()
        if len(msgs.messages) == 0 or st.button("Clear message history"):
            msgs.clear()


    # Setup Redis vector db retrieval
    retriever = configure_retriever(config.DOCS_FOLDER)

    # Configure Agent
    agent = configure_agent(chat_memory=msgs, tools=[retriever])

    # Chat Interface
    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        if msg.type in avatars:
            with st.chat_message(avatars[msg.type]):
                st.markdown(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything about the 2022 Chevy Colorado!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            response = generate_response(use_cache, llmcache, user_query, agent)
            st.markdown(response)
            if use_cache:
                # TODO - should we cache responses that were used from the cache?
                llmcache.store(user_query, response)


if __name__ == "__main__":
    render()
