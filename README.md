## 💬 Support Ticket Assistant (Enhanced)

A **Production-ready AI-powered chatbot** that intelligently classifies customer support queries and generates professional responses using company FAQs — built with **LangChain**, **Groq**, and **Streamlit**.

---

### 🚀 Features

#### Core Functionality
* 💡 **Intelligent Query Understanding** - Analyzes customer queries using company FAQs
* 🧠 **Context-Aware Conversations** - Remembers past conversations for better responses
* ⚙️ **Advanced Retrieval Chain** - Built with modern LangChain architecture
* 📊 **Real-time Classification** - Automatically categorizes, assesses urgency, and analyzes sentiment
* 💬 **Professional UI** - Modern, responsive Streamlit interface

#### Enhanced Features
* 🔧 **Modular Architecture** - Clean separation of concerns with config and utility modules
* 🛡️ **Robust Error Handling** - Comprehensive error management and validation
* 📈 **Live Status Monitoring** - Real-time system health indicators
* 🎯 **Sample Query Library** - Quick-start examples for common questions
* 📊 **Chat Analytics** - Track conversation statistics and metrics
* 🎫 **Ticket Classification Display** - Visual representation of query analysis
* 🔄 **Chat History Management** - Easy clearing and management of conversations

---

### 🧰 Tech Stack

* **LangChain v0.3+** – Modern retrieval and reasoning framework
* **Groq (Qwen-2.5-32B)** – High-performance LLM for fast responses
* **HuggingFace Embeddings** – Semantic text understanding
* **FAISS** – Efficient vector similarity search
* **Streamlit** – Interactive web interface
* **Python 3.8+** – Core runtime environment

---

### 🗂️ Enhanced Project Structure

```
📂 Support-Ticket-Assistant
│
├── 📄 app.py                 # Original Streamlit app (fixed)
├── 📄 app_enhanced.py        # Enhanced modular version
├── 📄 config.py              # Configuration management
├── 📄 utils.py               # Utility functions
├── 📄 test_utils.py          # Unit tests
├── 📄 FAQ.txt                # Company FAQ document
├── 📄 requirements.txt       # Updated dependencies
├── 📄 .env                   # Environment variables
└── 📘 README.md              # Enhanced documentation
```

---

### ⚡ Quick Start

#### Prerequisites
- Python 3.8 or higher
- Groq API key (get one at [console.groq.com](https://console.groq.com))

#### Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Support-Ticket-Assistant.git
cd Support-Ticket-Assistant

# 2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5️⃣ Run the application
# For basic version (fixed):
streamlit run app_fixed.py

# For enhanced version:
streamlit run app_enhanced.py

# Or use the runner script:
python run_app.py
```

---

### � Configuration

The application uses a modular configuration system:

#### Environment Variables (.env)
```bash
GROQ_API_KEY=your_groq_api_key_here
```

#### Configuration Options (config.py)
- `LLM_MODEL_NAME` - AI model to use (default: qwen-2.5-32b)
- `EMBEDDING_MODEL_NAME` - Text embedding model
- `CHUNK_SIZE` - Document chunk size for processing
- `RETRIEVAL_K` - Number of documents to retrieve
- And more...

---

### 🧪 Testing

Run the unit tests to verify functionality:

```bash
python test_utils.py
```

---

### � Usage Examples

#### Basic Query
* **User:** *"How can I reset my account password?"*
* **Category:** account
* **Urgency:** medium
* **Sentiment:** neutral
* **Response:** *"You can reset your account password by clicking on 'Forgot Password' on the login page and following the email verification process."*

#### Complex Query
* **User:** *"I was charged twice for order #12345 and haven't received my package yet!"*
* **Category:** billing
* **Urgency:** high
* **Sentiment:** negative
* **Response:** *"I sincerely apologize for the double charge and delivery issue. Please contact our support team with your order ID #12345, and we'll immediately process a refund for the duplicate charge and investigate your shipment status."*

---

### 🚀 Deployment

#### Local Development
```bash
streamlit run app_enhanced.py
```

#### Production (Docker)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app_enhanced.py"]
```

#### Cloud Platforms
- **Streamlit Cloud** - Direct deployment from GitHub
- **Heroku** - Using Docker container
- **AWS/Azure/GCP** - Container orchestration

---

### 🐛 Troubleshooting

#### Common Issues

1. **API Key Error**
   - Ensure GROQ_API_KEY is set in .env file
   - Verify the API key is valid and active

2. **Import Errors**
   - Run `pip install -r requirements.txt` to ensure all dependencies are installed
   - Check Python version compatibility (3.8+)

3. **FAQ File Not Found**
   - Ensure FAQ.txt exists in the project root
   - Check file permissions

4. **Memory Issues**
   - Reduce `CHUNK_SIZE` in config.py for large documents
   - Ensure sufficient system RAM

---

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

### 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### 👨‍💻 Author

**Paras Jain**
📧 [LinkedIn](www.linkedin.com/in/paras-jain-971000299)
🌐 [GitHub](https://github.com/parasjain08803)

---

### 🙏 Acknowledgments

- **LangChain Team** - For the amazing framework
- **Groq** - For providing fast, accessible AI models
- **Hugging Face** - For the embedding models
- **Streamlit** - For the excellent web app framework
