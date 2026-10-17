# AI Study Assistant — RAG Chatbot

An AI-powered study assistant that answers questions from uploaded PDF documents using Retrieval-Augmented Generation (RAG).

## 🚀 Live Demo

**Try the application:** [Add your deployed Streamlit URL here]

## ✨ Features

- Upload multiple PDF documents.
- Extract and chunk document text.
- Generate embeddings using Sentence Transformers.
- Retrieve relevant passages through semantic search.
- Generate context-grounded answers using Groq's LLM API.
- Handle follow-up questions using conversational history and query rewriting.
- Display source filenames and page numbers for retrieved passages.
- Use a Streamlit web interface.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **PDF Processing:** PyMuPDF
- **Embeddings:** Sentence Transformers
- **Numerical Computing:** NumPy
- **LLM API:** Groq
- **Version Control:** Git and GitHub

## 🧠 How It Works

1. The user uploads one or more PDF documents.
2. The application extracts text from each page.
3. The text is split into overlapping chunks while retaining source metadata.
4. Sentence Transformers converts the chunks into embeddings.
5. Semantic search retrieves passages relevant to the user's question.
6. Groq's language model generates an answer using the retrieved passages and conversation context.
7. The application displays the answer and the retrieved source passages.

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/theamankalwar/ai-study-assistant.git
cd ai-study-assistant
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Obtain your API key from [Groq Console](https://console.groq.com/keys).

**Never commit your `.env` file or expose your API key.**

### 5. Run the application

```bash
python -m streamlit run app.py
```

## ⚠️ Current Limitations

- Scanned PDFs are not supported unless text extraction or OCR is added.
- Retrieval currently uses a simple embedding-similarity approach.
- Generated answers may contain errors and should be checked against their source passages.
- API usage is subject to the provider's limits and policies.

## 🔮 Future Improvements

- Improve retrieval accuracy and document chunking.
- Add OCR support for scanned documents.
- Add more rigorous RAG evaluation.
- Improve page-level citations and answer verification.
- Add persistent chat history and document management.

## 👨‍💻 Author

**Aman Kumar**

- GitHub: [@theamankalwar](https://github.com/theamankalwar)

---

Built as a learning project to understand document processing, embeddings, semantic search, LLM integration, and Retrieval-Augmented Generation.
