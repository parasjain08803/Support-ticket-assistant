import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# -----------------------------
# Streamlit App Setup
# -----------------------------
st.set_page_config(page_title="💬 Support Ticket Assistant", page_icon="💬", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #3b82f6;'>💬 Support Ticket Assistant</h1>
    <p style='text-align: center; color: gray;'>Ask your questions about company policy, billing, or technical issues.</p>
""", unsafe_allow_html=True)

# -----------------------------
# Load and Prepare Data
# -----------------------------
@st.cache_resource
def load_retriever():
    loader = TextLoader("FAQ.txt", encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

retriever = load_retriever()

# -----------------------------
# LLM Setup
# -----------------------------
llm = ChatGroq(model_name="qwen/qwen3-32b")

# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are a professional AI support assistant.
Analyze the customer's query and the FAQ context, then output a structured response.

Context:
{context}

Customer query:
{input}

Please output in the following format:

User query: <repeat query here>
Category: <billing / technical / account / general / etc.>
Urgency: <low / medium / high>
Sentiment: <positive / neutral / negative>
Suggested response: <polite, clear reply to customer>
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "input"]
)

# -----------------------------
# Chain Setup (Stuff + Retrieval)
# -----------------------------
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# -----------------------------
# Chat Memory
# -----------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------
# Chat Interface
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("💬 Type your question here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = retrieval_chain.invoke({"input": user_query})
            llm_answer = result["answer"]

            def clean_response(text):
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", "", text)
                return text.strip()

            final_answer = clean_response(llm_answer)
            st.markdown(final_answer)

    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
