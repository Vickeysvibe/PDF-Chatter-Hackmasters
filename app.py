import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pickle
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


import os

with st.sidebar:
    st.title("PDF Query Bot 💬")
    st.markdown('''
    ## About : 
       This Chatbot analysis the PDF file, Youtube videos  and gives the perfect and best solutions to the questions within a short span of time. Using AI model likes Langchain, Streamlit and Chatgpt.
    
          
    ## Team :
    <ul>
        <li>Vigneshwaran</li>
        <li>Vijayavelu</li>
        <li>Sivaneshan</li>
        <li>Shuaib Mustafa</li>
        <li>Nithin</li>       
    </ul>
    <i style="display:block;height:330px;"></i>
     ''',unsafe_allow_html=True)
    st.header("Made my black squad")



def main():
    load_dotenv()
    st.title("Smart Bot 😎 ")
    pdf= st.file_uploader("upload your PDF files here",type="pdf")
    if pdf:
        pdf_read=PdfReader(pdf) 
        text=""
        for page in pdf_read.pages:
            text+=page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        file_name=pdf.name[:-4]

        if os.path.exists(f"{file_name}.pkl"):
            with open(f"{file_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
        st.title("Question : ")
        question=st.text_input(label="",placeholder="Enter your query from PDF")
        if question:
            similars=VectorStore.similarity_search(question,k=3)
            llm=OpenAI(temperature=0,model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm, chain_type="stuff")
            response=chain.run(input_documents=similars, question=question)
            st.title("Answer")
            st.markdown(f"<div style='font-size:1.5rem; background-color: #0f0f0f; padding: 10px; border-radius: 5px; border: solid 2px'>{response}</div>", unsafe_allow_html=True)
 
if __name__ == "__main__":
    main()
