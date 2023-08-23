import os
import streamlit as st
import uuid

from redisvl.llmcache.semantic import SemanticCache

from langchain.chat_models import ChatVertexAI, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.embeddings import VertexAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from dotenv import load_dotenv
load_dotenv()


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

# Setup Redis LLMCache
llmcache = SemanticCache(
    redis_url=os.environ["REDIS_URL"],
    threshold=0.75, # semantic similarity threshold
)

# Setup Redis memory for conversation history
msgs = RedisChatMessageHistory(
    session_id=st.session_state.session_id,
    url=os.environ["REDIS_URL"]
)

def reset_msg_history(msgs: RedisChatMessageHistory):
    msgs.clear()
    msgs.add_ai_message("I am a friendly AI assistant that can help you understand your Chevy 2022 Colorado vehicle based on the provided PDF car manual. Ask a question of your manual!")


st.set_page_config(page_title="Chat Your PDF", page_icon="📃")
st.title("📃 Chat Your PDF")

with st.sidebar:

    check_cache = st.checkbox("Use LLM cache?")

    if st.button("Clear LLM cache"):
        llmcache.clear()

    if len(msgs.messages) == 0 or st.button("Clear message history"):
        reset_msg_history(msgs)


@st.cache_resource()
def configure_retriever(path):
    # Read documents
    docs = []
    for file in os.listdir(path):
        print(file, flush=True)
        loader = PyPDFLoader(os.path.join(path, file))
        docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=10)
    splits = text_splitter.split_documents(docs)

    # Create embeddings and store in vectordb
    embeddings = VertexAIEmbeddings()
    vectordb = Redis.from_documents(splits, embeddings, redis_url=os.environ["REDIS_URL"], index_name="chatbot")

    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    return retriever


# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
#         self.container = container
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         print("new token", flush=True)
#         self.text += token
#         self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
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

def generate_response(check_cache, user_query, qachat) -> str:
    if check_cache:
        if response := llmcache.check(user_query):
            return response[0]

    retrieval_handler = PrintRetrievalHandler(st.container())
    # stream_handler = StreamHandler(st.empty())
    return qachat.run(user_query, callbacks=[retrieval_handler])


def render():
    retriever = configure_retriever(os.environ["DOCS_FOLDER"])
    memory = ConversationBufferMemory(
        memory_key="chat_history", chat_memory=msgs, return_messages=True
    )
    chatLLM = ChatVertexAI(
        temperature=0.1,
        project=os.environ["PROJECT_ID"],
        location=os.environ["LOCATION"]
    )
    # chatLLM = ChatOpenAI(streaming=True)
    print("making QA chain", flush=True)
    qachat = ConversationalRetrievalChain.from_llm(
        llm=chatLLM,
        memory=memory,
        retriever=retriever,
        verbose=True
    )

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        if msg.type in avatars:
            st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            print("GENERATING RESPONSE", flush=True)
            response = generate_response(check_cache, user_query, qachat)
            st.markdown(response)
            print("RESPONSE", response, flush=True)

            if check_cache:
                llmcache.store(user_query, response)


if __name__ == "__main__":
    render()