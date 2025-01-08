import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

load_dotenv()

st.title("Article Based Query Resolver AI Tool")

# Input fields for URLs
st.write("Enter the URLs you want to use for context:")
url_1 = st.text_input("URL 1", value='https://www.oracle.com/in/artificial-intelligence/machine-learning/what-is-machine-learning/')
url_2 = st.text_input("URL 2", value='https://www.geeksforgeeks.org/ml-machine-learning/')

process_urls_clicked = st.button("Load and Process URLs")

faiss_index_path = "faiss_index"

# Initialize embeddings outside the conditionals
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

if process_urls_clicked:
    # Load documents from URLs
    urls = [url_1, url_2]
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Create embeddings using GoogleGenerativeAIEmbeddings and save them to FAISS
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save the FAISS index to disk
    vectorstore.save_local(faiss_index_path)

    st.success("Documents processed and embeddings created successfully!")
    st.write("FAISS index saved to disk.")

# Input for querying the loaded documents
query = st.text_input("Ask a question:")
if query:
    if os.path.exists(faiss_index_path):
        st.write("FAISS index found. Loading...")

        try:
            # Load the FAISS index from disk with allow_dangerous_deserialization flag set to True
            vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            st.write("FAISS index loaded successfully.")

            # Create the QA chain with the retriever
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

            result = chain({"question": query}, return_only_outputs=True)

            # Display the answer and sources
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")
    else:
        st.warning("No FAISS index found. Please load URLs first.")
else:
    st.write("Please load URLs to start querying.")
