import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_unstructured import UnstructuredLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage



def get_document_loader(file_path: str, file_extension: str):

    if file_extension == ".pdf":
        return PyPDFLoader(file_path)
    elif file_extension in [".txt", ".docx"]:
        return UnstructuredLoader(file_path)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None



class LangChainRAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="./EmbeddingModel",  
            model_kwargs={"local_files_only": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.1
        )
        self.qa_chain = None

    def process_document(self, uploaded_file):
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            loader = get_document_loader(tmp_file_path, file_extension)
            if not loader: return 0
            
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            vectorstore = Chroma.from_documents(texts, self.embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,

            )
            return len(texts)

        finally:
            os.unlink(tmp_file_path)



def main():
    st.set_page_config(page_title="Chat With Your Documents", page_icon="💬")
    st.title("💬 Chat With Your Documents")
    st.markdown("Upload a PDF, TXT, or DOCX file and start asking questions!(wait for 2-3 minute to load fully)")



    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")

    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY not found! Add it to .env or .streamlit/secrets.toml.")
        st.stop()
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = LangChainRAGPipeline()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = None

    with st.sidebar:
        st.header("1. Upload Your Document")
        uploaded_file = st.file_uploader(
            "Supported formats: PDF, TXT, DOCX",
            type=["pdf", "txt", "docx"]
        )
        
        if uploaded_file and uploaded_file.name != st.session_state.get("processed_file"):
            with st.status("Processing document...", expanded=True):
                num_chunks = st.session_state.rag_pipeline.process_document(uploaded_file)
                st.session_state.processed_file = uploaded_file.name
                st.session_state.messages = []
            st.success(f"Document processed into {num_chunks} chunks. Ready to chat!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        if not st.session_state.processed_file:
            st.warning("Please upload and process a document first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.rag_pipeline.qa_chain:
                    

                    chat_history_for_chain = []

                    for msg in st.session_state.messages[:-1]: 
                        if msg["role"] == "user":
                            chat_history_for_chain.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history_for_chain.append(AIMessage(content=msg["content"]))


                    result = st.session_state.rag_pipeline.qa_chain.invoke({
                        "question": prompt,
                        "chat_history": chat_history_for_chain
                    })
                    response = result["answer"]
                    st.markdown(response)
                else:
                    response = "Error: QA chain not initialized."
                    st.error(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()