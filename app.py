import os
import re
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ========== 1. Load Environment ==========
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ========== 2. Load and Prepare FAQ ==========
loader = TextLoader("FAQ.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ========== 3. Initialize LLM ==========
llm = ChatGroq(model_name="qwen/qwen3-32b")

# ========== 4. Create Prompts ==========
reply_prompt_template = """
You are a helpful customer support assistant for a company.
Your job is to read the user query and the company FAQ context,
then generate a short, clear, professional reply.

Context from FAQ:
{context}

Customer query:
{question}

Your helpful reply:
"""

reply_prompt = PromptTemplate(
    template=reply_prompt_template,
    input_variables=["context", "question"]
)

# ========== 5. Memory ==========
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ========== 6. Create Chain ==========
conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": reply_prompt}
)

# ========== 7. Streamlit UI ==========
st.set_page_config(page_title="Support Ticket Assistant", page_icon="💬", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #3b82f6;'>💬 Support Ticket Assistant</h1>
    <p style='text-align: center; color: gray;'>Ask your questions</p>
""", unsafe_allow_html=True)


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["query"])

# User input box
user_query = st.chat_input("💬 Type your question here...")

if user_query:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Append to history
    st.session_state.chat_history.append({"role": "user", "query": user_query})

    # Show a typing spinner while waiting for response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = conversation_chain.invoke({"question": user_query})
            llm_answer = result["answer"]

            # Clean unwanted tags
            def clean_response(text):
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", "", text)
                return text.strip()

            answer = clean_response(llm_answer)
            st.markdown(answer)

    # Save assistant response
    st.session_state.chat_history.append({"role": "assistant", "query": answer})
