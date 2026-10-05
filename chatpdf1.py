import os
import streamlit as st

from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def get_pdf_text(pdf_docs):
    """
    Extract text from all uploaded PDF files.
    """

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

        except Exception as e:

            st.error(
                f"Error reading PDF '{pdf.name}': {e}"
            )

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def get_text_chunks(text):
    """
    Split large PDF text into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def get_vector_store(text_chunks):
    """
    Convert text chunks into Gemini embeddings
    and store them in FAISS.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

    # Convert chunks into LangChain Documents
    documents = [
        Document(page_content=chunk)
        for chunk in text_chunks
    ]

    # Create FAISS vector database
    vector_store = FAISS.from_documents(
        documents,
        embedding=embeddings
    )

    # Save vector database locally
    vector_store.save_local("faiss_index")


# ============================================================
# CREATE RAG PROMPT
# ============================================================

def get_prompt():

    prompt_template = """
You are a helpful assistant that answers questions based ONLY
on the provided PDF context.

Rules:

1. Use only the information provided in the context.
2. Do not use outside knowledge.
3. Do not make up information.
4. If the answer cannot be found in the context, respond exactly:
   "Answer is not available in the context."
5. Give a clear and concise answer.
6. Do not mention the context, retrieval, embeddings, or FAISS
   unless the user specifically asks about them.

Context:
{context}

Question:
{question}

Answer:
"""

    return ChatPromptTemplate.from_template(
        prompt_template
    )


# ============================================================
# LOAD GEMINI MODEL
# ============================================================

def get_llm():

    model = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=GOOGLE_API_KEY
    )

    return model


# ============================================================
# EXTRACT ONLY TEXT FROM GEMINI RESPONSE
# ============================================================

def extract_response_text(response):
    """
    Extract only the actual answer text from
    Gemini/LangChain AIMessage.

    Handles both:

    1. String response.content

    2. Structured response.content such as:

       [
           {
               "type": "text",
               "text": "Answer here...",
               "extras": {
                   "signature": "..."
               }
           }
       ]
    """

    content = response.content

    # --------------------------------------------------------
    # CASE 1: Normal string
    # --------------------------------------------------------

    if isinstance(content, str):

        return content.strip()

    # --------------------------------------------------------
    # CASE 2: Structured content blocks
    # --------------------------------------------------------

    if isinstance(content, list):

        text_parts = []

        for block in content:

            # Make sure block is a dictionary
            if isinstance(block, dict):

                # We only want blocks of type "text"
                if block.get("type") == "text":

                    text = block.get("text")

                    if text:
                        text_parts.append(str(text))

        return "\n".join(text_parts).strip()

    # --------------------------------------------------------
    # CASE 3: Unexpected format
    # --------------------------------------------------------

    return str(content).strip()


# ============================================================
# USER QUESTION
# ============================================================

def user_input(user_question):

    # --------------------------------------------------------
    # Check whether FAISS database exists
    # --------------------------------------------------------

    if not os.path.exists("faiss_index"):

        st.warning(
            "Please upload and process PDF files first."
        )

        return

    try:

        # ----------------------------------------------------
        # Create embedding model
        # ----------------------------------------------------

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=GOOGLE_API_KEY
        )

        # ----------------------------------------------------
        # Load FAISS vector database
        # ----------------------------------------------------

        new_db = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

        # ----------------------------------------------------
        # Retrieve relevant documents
        # ----------------------------------------------------

        docs = new_db.similarity_search(
            user_question,
            k=4
        )

        # ----------------------------------------------------
        # Check if documents were found
        # ----------------------------------------------------

        if not docs:

            st.warning(
                "No relevant information was found in the PDF."
            )

            return

        # ----------------------------------------------------
        # Combine retrieved documents
        # ----------------------------------------------------

        context = "\n\n".join(
            document.page_content
            for document in docs
        )

        # ----------------------------------------------------
        # Get Gemini model
        # ----------------------------------------------------

        model = get_llm()

        # ----------------------------------------------------
        # Get prompt
        # ----------------------------------------------------

        prompt = get_prompt()

        # ----------------------------------------------------
        # Create LangChain Runnable
        # ----------------------------------------------------

        chain = prompt | model

        # ----------------------------------------------------
        # Invoke chain
        # ----------------------------------------------------

        response = chain.invoke(
            {
                "context": context,
                "question": user_question
            }
        )

        # ----------------------------------------------------
        # Extract ONLY answer text
        # ----------------------------------------------------

        answer = extract_response_text(response)

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.write("### Reply:")

        if answer:

            st.markdown(answer)

        else:

            st.warning(
                "Gemini returned an empty response."
            )

    except Exception as e:

        st.error(
            f"Error while answering the question: {e}"
        )


# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main():

    st.header("Chat with PDF using Gemini 📚")

    st.write(
        "Upload one or more PDF files, process them, "
        "and ask questions about their contents."
    )

    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        st.title("📂 PDF Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        # ----------------------------------------------------
        # Process PDFs
        # ----------------------------------------------------

        if st.button(
            "Submit & Process",
            use_container_width=True
        ):

            if not pdf_docs:

                st.warning(
                    "Please upload at least one PDF file."
                )

            else:

                with st.spinner(
                    "Processing PDF files..."
                ):

                    # ----------------------------------------
                    # Extract PDF text
                    # ----------------------------------------

                    raw_text = get_pdf_text(
                        pdf_docs
                    )

                    if not raw_text.strip():

                        st.error(
                            "No readable text was found in "
                            "the uploaded PDF files."
                        )

                        return

                    # ----------------------------------------
                    # Split text into chunks
                    # ----------------------------------------

                    text_chunks = get_text_chunks(
                        raw_text
                    )

                    # ----------------------------------------
                    # Create FAISS vector database
                    # ----------------------------------------

                    get_vector_store(
                        text_chunks
                    )

                st.success(
                    "PDF processing completed successfully! ✅"
                )

                st.info(
                    f"Created {len(text_chunks)} text chunks."
                )

    # --------------------------------------------------------
    # Question input
    # --------------------------------------------------------

    st.subheader("💬 Ask a Question")

    user_question = st.text_input(
        "Ask a question from the PDF files:"
    )

    if user_question:

        user_input(
            user_question
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    main()
