import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

# --- NEW: Import libraries for local Hugging Face models ---
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- REMOVED: No longer need Google API libraries ---

# --- CORE FUNCTIONS ---

def get_pdf_text(pdf_docs):
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits a long text into smaller, manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a vector store from text chunks using a local model."""
    if not text_chunks:
        st.warning("No text found in PDFs to process.")
        return
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("PDFs processed and indexed successfully!")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# --- MAJOR CHANGE: Function to load a local LLM ---
@st.cache_resource
def get_llm_pipeline():
    """Loads a local language model and tokenizer for conversation."""
    model_id = "google/flan-t5-base" # A good, small model (~900MB)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512 # Controls the maximum length of the generated response
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

def get_conversational_chain():
    """Sets up the conversational chain using the local LLM."""
    prompt_template_str = """
    You are a helpful assistant. Use the conversation history to answer the question.

    Conversation History:
    {history}

    Question:
    {input}

    Answer:
    """
    
    # --- CHANGE: Use the local LLM pipeline ---
    model = get_llm_pipeline()
    
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["history", "input"])
    memory = ConversationBufferWindowMemory(k=5, memory_key="history", input_key="input")
    
    chain = ConversationChain(
        llm=model,
        prompt=prompt,
        memory=memory,
        verbose=True
    )
    return chain

def handle_userinput(user_question):
    """Handles user questions by augmenting the prompt and generating a response."""
    try:
        if not os.path.exists("faiss_index"):
            st.warning("Please process your documents first.")
            return

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        context = "\n".join([doc.page_content for doc in docs])
        augmented_input = f"""
        Answer the following question based on the provided context. If the answer is not in the context, say so.

        Context:
        {context}

        Question:
        {user_question}
        """

        chain = st.session_state.conversation
        response = chain.run(input=augmented_input)
        
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", response))
    except Exception as e:
        st.error(f"An error occurred: {e}")

# --- STREAMLIT UI ---

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    st.header("RAG-ChatBot :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversational_chain()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for speaker, message in st.session_state.chat_history:
        with st.chat_message("user" if speaker == "You" else "assistant"):
            st.write(message)
    
    user_question = st.chat_input("Ask a Question about your documents")
    if user_question:
        handle_userinput(user_question)
        st.rerun()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing... This may take a moment."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
            else:
                st.warning("Please upload at least one PDF file.")

if __name__ == "__main__":
    main()

