import streamlit as st
from rag_pipeline import setup_rag_chain, process_and_ingest_pdf

st.set_page_config(page_title="Chat with your Data", page_icon="🤖", layout="wide")
st.title("Chat with your Data 🤖")

# --- SIDEBAR: Document Upload & Controls ---
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Chunking, embedding, and uploading to Pinecone..."):
                try:
                    success = process_and_ingest_pdf(uploaded_file)
                    if success:
                        st.success("Document successfully indexed!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    
    st.divider()
    
    # New: Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message has sources, display them in an expander
        if "sources" in message and message["sources"]:
            with st.expander("View Source Context"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Chunk {i+1}:**\n {doc.page_content}")
                    st.divider()

# Handle new user input
if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # 1. Display and save user prompt
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Setup the streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        
        try:
            # Get the chain
            chain = setup_rag_chain()
            
            # Stream the response chunk by chunk
            for chunk in chain.stream({"input": prompt}):
                # Extract the answer tokens as they arrive
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    message_placeholder.markdown(full_response + "▌")
                
                # Extract the source documents (this usually arrives in the first chunk)
                if "context" in chunk:
                    sources = chunk["context"]

            # Finalize the display
            message_placeholder.markdown(full_response)
            
            # Display sources in an expander for the current response
            if sources:
                with st.expander("View Source Context"):
                    for i, doc in enumerate(sources):
                        # Extract page numbers if the PDF loader grabbed them
                        page_num = doc.metadata.get('page', 'N/A')
                        st.markdown(f"**Source (Page {page_num}):**\n {doc.page_content}")
                        st.divider()

        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            message_placeholder.markdown(full_response)

    # 3. Save the assistant's response and sources to session state
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "sources": sources
    })