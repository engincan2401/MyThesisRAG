import streamlit as st
import os
import shutil
import time
import uuid 


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Студентски Асистент", page_icon="🎓", layout="wide")

# ПАПКИ
TEMP_FOLDER = "temp_data"

# Вместо една постоянна папка, ще ползваме сесията, за да помним къде е текущата база
if "db_path" not in st.session_state:
    st.session_state.db_path = "vector_db_current"


def reset_system():
    """Изчиства всичко, за да започнем на чисто"""
    st.cache_resource.clear()
    
    # Трием старата папка с базата, ако съществува
    if os.path.exists(st.session_state.db_path):
        try:
            shutil.rmtree(st.session_state.db_path)
        except:
            pass # Ако не може да се изтрие, просто ще направим нова с ново име
            
    # Генерираме ново уникално име за папката на базата
    # Така гарантираме, че няма да чете стари данни!
    new_id = str(uuid.uuid4())[:8]
    st.session_state.db_path = f"vector_db_{new_id}"
    
    # Трием временните файлове
    if os.path.exists(TEMP_FOLDER):
        try:
            shutil.rmtree(TEMP_FOLDER)
        except:
            pass
    os.makedirs(TEMP_FOLDER)

def save_uploaded_files(uploaded_files):
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def create_vector_db(files):
    documents = []
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    total_files = len(files)
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        status_text.info(f"📄 Четене на файл: {filename}...")
        
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                
        except Exception as e:
            st.error(f"Грешка при {filename}: {e}")
        
        progress_bar.progress((i + 1) / total_files)

    if not documents:
        st.error("❌ Не беше намерен текст във файловете!")
        return False

    status_text.info(f"✂️ Нарязване на {len(documents)} страници текст...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    status_text.info(f"🧠 Генериране на вектори в папка {st.session_state.db_path}...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Създаваме новата база в уникалната папка
    Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=st.session_state.db_path)
    
    status_text.success("✅ Готово! Базата е обновена.")
    progress_bar.empty()
    time.sleep(1)
    status_text.empty()
    return True

@st.cache_resource
def get_rag_chain(current_db_path):
    if not os.path.exists(current_db_path):
        return None
    
    llm = OllamaLLM(model="mistral")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    db = Chroma(persist_directory=current_db_path, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5}) 
    
    system_prompt = (
        "Ти си полезен асистент. Използвай САМО следния контекст, за да отговориш. "
        "Ако отговорът го няма в контекста, кажи 'Няма информация в качените файлове'. "
        "Не си измисляй факти.\n\n"
        "Контекст: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

#UI 

st.title("🎓 RAG Асистент (Чиста версия)")

with st.sidebar:
    st.header("1. Качи файлове")
    uploaded_files = st.file_uploader("Word или PDF", type=["pdf", "docx"], accept_multiple_files=True)
    
    if st.button("2. СЪЗДАЙ НОВА БАЗА"):
        if uploaded_files:
            reset_system()
            
            with st.spinner("Обработка..."):
                file_paths = save_uploaded_files(uploaded_files)
                success = create_vector_db(file_paths)
                
                if success:
                    st.rerun()
        else:
            st.warning("Няма избрани файлове!")
            
    st.markdown("---")
    st.write(f"Текуща база: `{st.session_state.db_path}`")

# Логика за чата
if not os.path.exists(st.session_state.db_path):
    st.info("Моля, качи файлове и натисни бутона 'СЪЗДАЙ НОВА БАЗА'!")
else:
    # Зареждаме веригата с конкретната папка
    rag_chain = get_rag_chain(st.session_state.db_path)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Питай нещо от файла..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Търся..."):
                try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Грешка: {e}")