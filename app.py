import os
import re
import streamlit as st
from dotenv import load_dotenv

# ✅ LangChain (v0.3+) imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain



# -------------------------------------------------------
# 🔑 Load environment variables
# -------------------------------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# -------------------------------------------------------
# ⚙️ Load and index documents (cached)
# -------------------------------------------------------
@st.cache_resource(show_spinner="Indexing FAQ...")
def load_vectorstore():
    try:
        loader = TextLoader("FAQ.txt", encoding="utf-8")
        documents = loader.load()
    except Exception as e:
        st.error(f"❌ Error loading FAQ.txt: {e}")
        st.stop()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 3})


retriever = load_vectorstore()


# -------------------------------------------------------
# 🧠 Initialize LLM (Groq)
# -------------------------------------------------------
try:
    llm = ChatGroq(model_name="qwen/qwen3-32b")  # ✅ Stable Groq model
except Exception as e:
    st.error(f"❌ Error initializing LLM: {e}")
    st.stop()


# -------------------------------------------------------
# 💬 Prompt Template
# -------------------------------------------------------
reply_prompt_template = """
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

reply_prompt = PromptTemplate(
    template=reply_prompt_template,
    input_variables=["context", "input"]
)



# -------------------------------------------------------
# 🧩 Build retrieval + memory chain
# -------------------------------------------------------
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
memory = st.session_state.memory

# Step 1: Create document chain (stuff)
document_chain = create_stuff_documents_chain(llm, reply_prompt)

# Step 2: Make retriever aware of conversation history
history_aware_retriever = create_history_aware_retriever(
    llm=llm,
    retriever=retriever,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "Use the chat history to better understand follow-up questions."),
        ("human", "{input}")
    ])
)

# Step 3: Create full retrieval chain
retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)


# -------------------------------------------------------
# 🖥️ Streamlit UI
# -------------------------------------------------------
st.set_page_config(page_title="Support Ticket Assistant", page_icon="💬", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #3b82f6;'>💬 Support Ticket Assistant</h1>
    <p style='text-align: center; color: gray;'>Ask your questions about company policy, billing, or technical issues.</p>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("💬 Type your question here...")


# -------------------------------------------------------
# 🤖 Chat handling
# -------------------------------------------------------
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = retrieval_chain.invoke({"input": user_query, "chat_history": memory.chat_memory})
                llm_answer = result["answer"]
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.stop()

            # Clean LLM output
            def clean_response(text):
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", "", text)
                return text.strip()

            final_answer = clean_response(llm_answer)
            st.markdown(final_answer)

    st.session_state.chat_history.append({"role": "assistant", "content": final_answer})
