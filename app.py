from dotenv import load_dotenv
import openai
import os
import streamlit as st
from PyPDF2 import PdfReader #  import pdf reader
from langchain.text_splitter import CharacterTextSplitter # import text splitter

from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS
# from langchain.document_loaders import PyPDFLoader
from langchain.llms import AzureOpenAI
from langchain.chains.question_answering import load_qa_chain

# load_dotenv('.env')
def main():
    load_dotenv()

   # Configure Azure OpenAI Service API
    openai.api_key = os.getenv("OPENAI_API_KEY") # set OpenAI API key
    # print(f"OPENAI_API_KEY:{openai.api_key}") # print OpenAI API key
    openai.api_base = os.getenv("OPENAI_API_BASE") # set OpenAI Base URL
    # print(f"OPENAI_API_BASE:{openai.api_base}") # print OpenAI Base URL
    
    openai.api_type = os.getenv("OPENAI_API_TYPE")
    openai.api_version = os.getenv("OPENAI_API_VERSION")
    # print(f"OPENAI_API_VERSION:{openai.api_version}") # print OpenAI Base URL


    st.set_page_config(
        page_title="Chat 💬 with your PDF 📄",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="auto",
    )
    st.header("Talk to your PDF 💬")

    # Upload PDF file
    pdf = st.file_uploader("Upload PDF 📑")

     # check if user has uploaded a file
    # extract the text from the pdf file
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = "" # empty string to store the text
        for page in pdf_reader.pages: # for each page in the pdf
            text += page.extract_text() # concatenating all the text from the pages
   
            # # display the text on the streamlit app
            # st.write(text)

        # Use function from langchain to split the text into chunks
        text_splitter = CharacterTextSplitter(
            separator = "\n", # split by new line
            chunk_size = 1000, # split into chunks of 1000 characters
            chunk_overlap = 200, # overlap chunks by 200 characters
            length_function =  len # use the len function to get the length of the chunk
        )
        text_chunks = text_splitter.split_text(text)
        # st.write(chunks) # display the text on the streamlit app  

    #     # Define models
        model = "text-embedding-ada-002"
        completion_model = "text-davinci-003"

        # Create embeddings
        embeddings = OpenAIEmbeddings(model=model,
                                      deployment=model,
                                      openai_api_key= openai.api_key, 
                                      chunk_size=1)
        st.write("Embeddings created")

        with st.spinner("It's indexing..."):
            #  Create a knowledge base using FAISS from the given chunks and embeddings
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.success("Embeddings done.", icon="✅")

        # get user question
        question = st.text_input("Ask your question here:") # get user question

        if question: # if user question is not empty
            # search for similar documents
            docs_db = vectorstore.similarity_search(question)
            
            # Define the LLM model
            llm = AzureOpenAI(deployment_name=completion_model, 
                              model_name=completion_model,
                              temperature=0.5,
                              max_tokens=2000) 
            chain = load_qa_chain(llm, chain_type="stuff")

            # Send the question and the documents to the LLM model
            response = chain({"input_documents": docs_db,
                              "question": question,
                              "language": "English",
                              "existing_answer" : ""},
                              return_only_outputs=True)
            
            with st.spinner("It's thinking..."):
                st.write(response) # display the response on the streamlit app
       
if __name__ == '__main__':
    main()