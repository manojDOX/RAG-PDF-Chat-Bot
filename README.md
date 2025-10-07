# 🤖 RAG Chatbot for PDF Documents  
> *"Chat with your PDFs using local, open-source AI models — completely private and offline."*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)
![LangChain](https://img.shields.io/badge/AI%20Pipeline-LangChain-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

---

## 🧩 Overview  
The **RAG Chatbot for PDF Documents** is an **interactive web application** that lets you *chat with your PDF files*.  
Upload multiple PDFs, and the app will use a **local, open-source language model** to answer your questions — with all processing happening *entirely on your device* for full privacy.  

This project leverages the **Retrieval-Augmented Generation (RAG)** technique, ensuring that responses are **contextually accurate**, **relevant**, and **grounded** in your own documents.  

---

## ✨ Features  

✅ **Chat with Multiple PDFs** – Upload one or more PDFs to form a unified knowledge base.  
🔒 **Local & Private** – All computations run on your local system. No data leaves your device.  
🧠 **Open-Source Models** – Powered by free Hugging Face models for embeddings and generation.  
💬 **Interactive UI** – Simple, modern, and intuitive interface built with Streamlit.  

---

## ⚙️ How It Works  

### 🔍 1. Processing Phase  
1. **Text Extraction:** Extracts text from your uploaded PDFs.  
2. **Chunking:** Splits large text into smaller, context-friendly chunks.  
3. **Embedding & Indexing:**  
   - Each chunk is embedded using `all-MiniLM-L6-v2`.  
   - Embeddings are stored in a **FAISS** vector database for fast similarity search.  

### 💬 2. Question-Answering Phase  
1. **Similarity Search:** Converts your query into a vector and finds relevant chunks.  
2. **Prompt Augmentation:** Combines retrieved chunks with your query for context.  
3. **Answer Generation:**  
   - Uses `google/flan-t5-base` (local model) to generate precise answers.  

---

## 🧰 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| **Frontend / UI** | Streamlit |
| **AI Orchestration** | LangChain |
| **Language Model (LLM)** | `google/flan-t5-base` |
| **Embedding Model** | `all-MiniLM-L6-v2` |
| **Vector Store** | FAISS |
| **PDF Processing** | PyPDF2 |
| **Backend Language** | Python 3.8+ |

---

## 🚀 Setup & Installation  

### 1️⃣ Clone the Repository  
```bash
git clone <your-repository-url>
cd <your-repository-folder>
````

### 2️⃣ Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Once setup is complete, launch the app:

```bash
streamlit run app.py
```

💡 *Note:* The first run may take a few minutes as it downloads pre-trained models (~900MB).
Subsequent runs will be much faster.

---

## 🧠 Usage Guide

1. **Upload PDFs:**

   * Use the sidebar to upload one or more PDF files.

2. **Process Documents:**

   * Click **“Process”** to extract, embed, and index your PDFs.
   * Wait for the *success message* confirming processing completion.

3. **Ask Questions:**

   * Type your question in the chat box.
   * Receive accurate, document-grounded answers in seconds!

---

## 🧩 Architecture Diagram

```
📄 PDFs  
   ↓  
🧱 Text Extraction → Chunking → Embedding (MiniLM) → FAISS Index  
   ↓  
❓ User Query → Vector Search → Prompt Augmentation → LLM (Flan-T5) → 🧠 Answer
```

---

## 🛠️ Future Enhancements

* [ ] Add support for DOCX and TXT files
* [ ] Integrate more local LLMs (e.g., Mistral, Llama 3)
* [ ] Enable persistent chat memory
* [ ] Include dark/light UI themes

---

## 📜 License

This project is licensed under the **MIT License** – free to use, modify, and distribute.

---

## 💡 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [Hugging Face Transformers](https://huggingface.co/)
* [Streamlit](https://streamlit.io/)
* [FAISS](https://github.com/facebookresearch/faiss)

---

### ⭐ If you like this project, don’t forget to give it a **star** on GitHub!

> “Built with ❤️ and powered by open-source intelligence.”

```
