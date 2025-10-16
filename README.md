Perfect 👍 Here’s a **clean, professional, and company-ready** `README.md` for your project — short, attractive, and easy to understand:

---

## 💬 Support Ticket Assistant

A **Generative AI-powered chatbot** that helps classify customer support queries and generate professional responses using company FAQs — built with **LangChain**, **Groq**, and **Streamlit**.

---

### 🚀 Features

* 💡 Understands and replies to customer queries using company FAQs
* 🧠 Remembers past conversations for context-aware answers
* ⚙️ Built with advanced **Conversational Retrieval Chain**
* 💬 Interactive, professional **Streamlit UI**
* ⚡ Uses **Groq Qwen3-32B LLM** for fast and accurate responses

---

### 🧰 Tech Stack

* **LangChain** – for building retrieval and reasoning logic
* **Groq (Qwen3-32B)** – LLM for generating human-like responses
* **HuggingFace Embeddings** – for semantic understanding of text
* **FAISS** – for efficient document similarity search
* **Streamlit** – for building the web interface

---

### 🗂️ Project Structure

```
📂 Support-Ticket-Assistant
│
├── 📄 app.py                # Main Streamlit app
├── 📄 FAQ.txt               # Company FAQ document
├── 📄 requirements.txt      # Project dependencies
└── 📘 README.md             # Project documentation
```

---

### ⚡ How to Run Locally

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Support-Ticket-Assistant.git
cd Support-Ticket-Assistant

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Add your GROQ API key in .env
GROQ_API_KEY=your_api_key_here

# 4️⃣ Run the Streamlit app
streamlit run app.py
```

---

### 👨‍💻 Author

**Paras Jain**
📧 [LinkedIn](https://www.linkedin.com/in/parasjain)
🌐 [GitHub](https://github.com/parasjain)

---

### 💭 Example Query

**User query:** *“How can I reset my account password?”*
**Category:** Account Management
**Urgency:** Medium
**Sentiment:** Neutral
**Suggested response:** *You can reset your account password by clicking on “Forgot Password” on the login page and following the email verification process.*

