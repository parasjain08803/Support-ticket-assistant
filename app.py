import streamlit as st
import os
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Load environment variables
load_dotenv()

# Simple Support Ticket Assistant
class SimpleTicketAssistant:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.faq_file = "FAQ.txt"
        self.retriever = None
        self.chain = None
        
    def initialize(self):
        """Initialize the assistant components"""
        try:
            # Load FAQ documents
            if not os.path.exists(self.faq_file):
                st.error(f"FAQ file not found: {self.faq_file}")
                return False
                
            loader = TextLoader(self.faq_file, encoding="utf-8")
            documents = loader.load()
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, 
                chunk_overlap=100
            )
            docs = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vectorstore = FAISS.from_documents(docs, embeddings)
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Initialize LLM
            self.llm = ChatGroq(
                model_name="qwen/qwen3-32b",
                api_key=self.groq_api_key
            )
            
            # Create base prompt template
            self.base_prompt = """You are a helpful customer support assistant. Analyze customer's question and provide a structured response based on the provided context.

{conversation_context}Context from knowledge base:
{context}

Question:
{input}

Provide your response in this exact format:

Category: [billing/technical/account/general/shipping/returns/offers]
Urgency: [low/medium/high]
Sentiment: [positive/neutral/negative]
Suggested Response: [Your helpful response here]"""
            
            # Create chains
            document_chain = create_stuff_documents_chain(self.llm, PromptTemplate(
                template=self.base_prompt.replace("{conversation_context}", ""),
                input_variables=["context", "input"]
            ))
            self.chain = create_retrieval_chain(self.retriever, document_chain)
            
            return True
            
        except Exception as e:
            st.error(f"Error initializing assistant: {e}")
            return False
    
    def process_query(self, query, chat_history=None):
        """Process user query with memory"""
        try:
            if not self.chain:
                return "AI system is not initialized. Please check configuration."
            
            # Create conversation context from chat history
            conversation_context = ""
            if chat_history:
                conversation_context = "Previous conversation:\n"
                for msg in chat_history[-6:]:  # Last 6 messages for context
                    if msg["role"] == "user":
                        conversation_context += f"User: {msg['content']}\n"
                    else:
                        # Extract only suggested response from previous AI messages
                        resp_match = re.search(r'\*\*Suggested Response:\*\*\n(.+)', msg["content"], re.DOTALL)
                        if resp_match:
                            conversation_context += f"Assistant: {resp_match.group(1).strip()}\n"
                        else:
                            conversation_context += f"Assistant: {msg['content']}\n"
                conversation_context += "\n"
            
            # Create temporary prompt with memory
            temp_prompt = PromptTemplate(
                template=self.base_prompt.replace("{conversation_context}", conversation_context),
                input_variables=["context", "input"]
            )
            
            # Create temporary chain with memory-enhanced prompt
            document_chain = create_stuff_documents_chain(self.llm, temp_prompt)
            temp_chain = create_retrieval_chain(self.retriever, document_chain)
            
            result = temp_chain.invoke({"input": query})
            response = result.get("answer", "I apologize, but I couldn't generate a response. Please try again.")
            
            # Parse structured response
            structured_response = self.parse_structured_response(response)
            return structured_response
            
        except Exception as e:
            return f"Error processing your query: {e}"
    
    def parse_structured_response(self, response):
        """Parse the structured response into components"""
        import re
        
        # Extract category
        category_match = re.search(r'Category:\s*(\w+)', response, re.IGNORECASE)
        category = category_match.group(1).strip() if category_match else "general"
        
        # Extract urgency
        urgency_match = re.search(r'Urgency:\s*(\w+)', response, re.IGNORECASE)
        urgency = urgency_match.group(1).strip() if urgency_match else "medium"
        
        # Extract sentiment
        sentiment_match = re.search(r'Sentiment:\s*(\w+)', response, re.IGNORECASE)
        sentiment = sentiment_match.group(1).strip() if sentiment_match else "neutral"
        
        # Extract suggested response
        response_match = re.search(r'Suggested Response:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        suggested_response = response_match.group(1).strip() if response_match else response
        
        return {
            "category": category,
            "urgency": urgency,
            "sentiment": sentiment,
            "suggested_response": suggested_response
        }

def export_chat_history():
    """Export chat history to CSV"""
    
    if "messages" not in st.session_state or not st.session_state.messages:
        st.warning("No chat history to export")
        return
    
    # Prepare data for export
    export_data = []
    current_ticket = {}
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            current_ticket["user_query"] = msg["content"]
            current_ticket["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif msg["role"] == "assistant" and "Category:" in msg["content"]:
            # Parse structured response
            cat_match = re.search(r'\*\*Category:\*\* (\w+)', msg["content"])
            urg_match = re.search(r'\*\*Urgency:\*\* (\w+)', msg["content"])
            sent_match = re.search(r'\*\*Sentiment:\*\* (\w+)', msg["content"])
            resp_match = re.search(r'\*\*Suggested Response:\*\*\n(.+)', msg["content"], re.DOTALL)
            
            current_ticket["category"] = cat_match.group(1) if cat_match else "general"
            current_ticket["urgency"] = urg_match.group(1) if urg_match else "medium"
            current_ticket["sentiment"] = sent_match.group(1) if sent_match else "neutral"
            current_ticket["ai_response"] = resp_match.group(1).strip() if resp_match else msg["content"]
            
            export_data.append(current_ticket.copy())
            current_ticket = {}
    
    # Create CSV
    if export_data:
        csv_content = "Timestamp,User Query,Category,Urgency,Sentiment,AI Response\n"
        for ticket in export_data:
            csv_content += f'"{ticket["timestamp"]}","{ticket["user_query"]}","{ticket["category"]}","{ticket["urgency"]}","{ticket["sentiment"]}","{ticket["ai_response"]}"\n'
        
        st.download_button(
            label="📥 Download CSV",
            data=csv_content,
            file_name=f"support_tickets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        st.success("Chat history exported successfully!")
    else:
        st.warning("No structured tickets found to export")

def main():
    """Main application"""
    st.set_page_config(
        page_title="Support Ticket Assistant",
        page_icon="💬",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings & Tools")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        # Sample queries
        st.subheader("� Sample Queries")
        sample_queries = [
            "How can I reset my password?",
            "What's your return policy?",
            "I was charged twice for my order",
            "When will my order arrive?",
            "My coupon code isn't working",
            "I received a damaged item",
            "How do I cancel my order?",
            "Billing inquiry",
            "Technical support needed",
            "Account access issue"
        ]
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}"):
                st.session_state.temp_query = query
                st.rerun()
        
        # Statistics
        if "messages" in st.session_state and st.session_state.messages:
            st.subheader("📊 Statistics")
            
            # Count tickets by category
            categories = {}
            
            for msg in st.session_state.messages:
                if msg["role"] == "assistant" and "Category:" in msg["content"]:
                    # Extract category
                    cat_match = re.search(r'\*\*Category:\*\* (\w+)', msg["content"])
                    if cat_match:
                        cat = cat_match.group(1)
                        categories[cat] = categories.get(cat, 0) + 1
            
            if categories:
                st.write("**Categories:**")
                for cat, count in categories.items():
                    st.write(f"• {cat}: {count}")
            
            total_tickets = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("Total Tickets", total_tickets)
        
        # Export functionality
        st.subheader("📤 Export")
        if st.button("Export Chat History"):
            export_chat_history()
    
    # Main content
    st.title("💬 Support Ticket Assistant")
    st.markdown("AI-powered customer support with intelligent ticket classification and response generation.")
    
    # Initialize assistant
    if "assistant" not in st.session_state:
        with st.spinner("Initializing AI assistant..."):
            assistant = SimpleTicketAssistant()
            if assistant.initialize():
                st.session_state.assistant = assistant
                st.success("AI Assistant ready!")
            else:
                st.error("Failed to initialize AI Assistant")
                st.stop()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Always show chat input
    prompt = st.chat_input("Type your question here...")
    
    # Check for temp query from sidebar
    if "temp_query" in st.session_state:
        prompt = st.session_state.temp_query
        del st.session_state.temp_query
    
    # Process prompt if available
    if prompt:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Get chat history for memory (exclude current user message)
                chat_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 0 else []
                response = st.session_state.assistant.process_query(prompt, chat_history)
                
                # Display structured output
                if isinstance(response, dict):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Category", response["category"])
                    with col2:
                        st.metric("Urgency", response["urgency"])
                    with col3:
                        st.metric("Sentiment", response["sentiment"])
                    
                    st.markdown("**Suggested Response:**")
                    st.markdown(response["suggested_response"])
                    
                    # Format for chat history
                    formatted_response = f"**Category:** {response['category']}\n**Urgency:** {response['urgency']}\n**Sentiment:** {response['sentiment']}\n\n**Suggested Response:**\n{response['suggested_response']}"
                else:
                    st.markdown(response)
                    formatted_response = response
        
        # Add assistant message
        st.session_state.messages.append({"role": "assistant", "content": formatted_response})

if __name__ == "__main__":
    main()
