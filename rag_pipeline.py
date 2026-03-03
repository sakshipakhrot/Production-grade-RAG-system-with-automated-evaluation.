import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

def process_and_ingest_pdf(uploaded_file):
    """Saves uploaded file temporarily, chunks it, and pushes to Pinecone."""
    # 1. Save uploaded file to a temporary file on disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    try:
        # 2. Load and Chunk the text
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        # 3. Embed and Upsert to Vector DB
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index_name = os.environ.get("PINECONE_INDEX_NAME")
        
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name
        )
        return True
    
    finally:
        # 4. Mandatory Cleanup: Delete the temp file to save storage/memory
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def setup_rag_chain():
    # Setup Embeddings & Vector Store connections
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = os.environ.get("PINECONE_INDEX_NAME")
    
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Enable streaming on the LLM
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant", streaming=True)

    system_prompt = (
        "You are a helpful and precise assistant. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Keep the answer concise and professional.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

def get_answer(question: str) -> dict:
    chain = setup_rag_chain()
    
    response = chain.invoke({"input": question})
    
    return {
        "answer": response["answer"],
        "contexts": [doc.page_content for doc in response["context"]] # Extract text from chunks

    }
