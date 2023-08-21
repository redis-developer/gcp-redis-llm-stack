import os
import streamlit as st
import vertexai
import uuid

from langchain.chat_models import ChatVertexAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain.embeddings import VertexAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.messages import SystemMessage

# from redisvl.llmcache import SemanticCache

from dotenv import load_dotenv
load_dotenv()


vertexai.init(project=os.environ['PROJECT_ID'], location=os.environ['LOCATION'])


if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

st.set_page_config(page_title="Chat Your PDF", page_icon="📃")
st.title("📃 Chat Your PDF")


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
    vectordb = Redis.from_documents(splits, embeddings, redis_url=os.environ["REDIS_URL"])

    # Define retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# class PrintRetrievalHandler(BaseCallbackHandler):
#     def __init__(self, container):
#         self.container = container.expander("Context Retrieval")

#     def on_retriever_start(self, query: str, **kwargs):
#         self.container.write(f"**Question:** {query}")

#     def on_retriever_end(self, documents, **kwargs):
#         # self.container.write(documents)
#         for idx, doc in enumerate(documents):
#             source = os.path.basename(doc.metadata["source"])
#             self.container.write(f"**Document {idx} from {source}**")
#             self.container.markdown(doc.page_content)


def reset_msg_history(msgs: RedisChatMessageHistory):
    msgs.clear()
    msgs.add_message(SystemMessage(content="""You are a friendly AI assistant that can help a user understand their chosen PDF document.
                                   They will ask questions about the document in question and you can respond with known information driven by facts alone.
                                   You may not discuss arbitrary topics or veer too far off course in converation."""))
    msgs.add_ai_message("How can I help you?")


def render():
    # openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.")
    #     st.stop()

    retriever = configure_retriever(os.environ["DOCS_FOLDER"])

    # Setup memory for contextual conversation
    msgs = RedisChatMessageHistory(
        session_id=st.session_state.session_id,
        url=os.environ["REDIS_URL"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Setup LLM and QA chain
    llm = ChatVertexAI(
        temperature=0.1, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True
    )

    # # Setup LLMCache
    # llmcache = SemanticCache(
    #     redis_url=os.environ["REDIS_URL"],
    #     threshold=0.9, # semantic similarity threshold
    # )

    if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
        reset_msg_history(msgs)

    avatars = {"human": "user", "ai": "assistant"}
    for msg in msgs.messages:
        if msg.type in avatars:
            st.chat_message(avatars[msg.type]).write(msg.content)

    if user_query := st.chat_input(placeholder="Ask me about your pdf!"):
        st.chat_message("user").write(user_query)

        with st.chat_message("assistant"):
            # retrieval_handler = PrintRetrievalHandler(st.container())
            stream_handler = StreamHandler(st.empty())
            # if result := llmcache.check(user_query):
            #     for token in result[0].split(" "):
            #         stream_handler.on_llm_new_token(token)
            # else:
            response = qa_chain.run(user_query, callbacks=[stream_handler])


if __name__ == "__main__":
    render()