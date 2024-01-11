import streamlit as st
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS 
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from html_taslak import css, bot_template,user_template


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf) 
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
        
def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=1000,
              chunk_overlap=200,
                separators="\n",
            length_function=len,
            )
        chunks=text_splitter.split_text(text)
        return chunks
    
# https://huggingface.co/hkunlp/instructor-xl   <------ Text embedding için iyi model 
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #şimdilik huggingface dursun 4gb pytorch modeli yüklüyor. uzun hikaye
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts= text_chunks, embedding = embeddings)
    return vectorstore    


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(kullanıcı_sorusu):
    response = st.session_state.conversation({"question":kullanıcı_sorusu})
    
    bot_response_html = bot_template.replace("{{MSG}}", response["answer"])
    st.markdown(bot_response_html, unsafe_allow_html=True)
    #st.write(response["answer"])
    
    user_question_html = user_template.replace("{{MSG}}", f"<p style='color:blue;'>{kullanıcı_sorusu}</p>")
    st.markdown(user_question_html, unsafe_allow_html=True)
    
    
    

def main():
    load_dotenv()
    st.set_page_config(page_title="PDF ile Eğitilebilir ChatBot,", page_icon=":books:")
    
    st.write(css,unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("Ne Öğrenmek İstiyorsun :books:")
    kullanıcı_sorusu = st.text_input("Ders notlarından anlamadığını veya takıldığın bir soruyu bana sorabilirsin:")
    if kullanıcı_sorusu:
        handle_userinput(kullanıcı_sorusu)
    st.write(user_template.replace("{{MSG}}","Selam ben de öğrenci falan filan"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Selam ben whizywhiz"),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Dökümanların")
        pdf_docs=st.file_uploader(
            "PDF Yükle ve Başla'ya Bas", accept_multiple_files=True)
        if st.button("Başla"):
            with st.spinner("Dosya yükleniyor"):
                # PDf text'ini al
                raw_text=get_pdf_text(pdf_docs)
                
                #cunksları oluşturma fonsksk
                chunks_text = get_text_chunks(raw_text)
                
                #vector storeları oluştran fonkss.
                vectorstore = get_vectorstore(chunks_text)

                ## conversation chaiinn
                st.session_state.conversation = get_conversation_chain(vectorstore)
            st.success('Dosya Yüklendi !')

if __name__ == "__main__":
    main()