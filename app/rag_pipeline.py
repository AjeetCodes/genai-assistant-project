from config import Config
from app_logger import AppLogger
import logging
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import asyncio
class RagPipline:
    pdfFileContent = ""
    def __init__(self):
        self.configObj = Config()
        self.loggerObj = AppLogger("rag_pipleline", logging.INFO).setupLogger()
        
    def loadPDFDoc(self, pdfFile):
        folderPath = os.path.join(self.configObj.ROOT_PATH, 'data')
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(self.configObj.ROOT_PATH, 'data', 'temp.pdf')
        self.loggerObj.info(f"file{filePath}")
        with open(filePath, "wb") as file:
            file.write(pdfFile.read())
            loader = PyMuPDFLoader(filePath)
            pages = loader.load()
            # self.loggerObj.info(f"page content {str(pages[0].page_content)}")
            for page in pages:
                self.pdfFileContent += page.page_content
            self.loggerObj.info(f"file content {str(self.pdfFileContent)}")
            self.embeddingsTexts()
    def embeddingsTexts(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunk = text_splitter.split_text(self.pdfFileContent)
        embiddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectore_store = Chroma.from_texts(
            texts=chunk,
            embedding=embiddings,
            persist_directory='./chroma_store'
        )
        res = vectore_store.persist()
    def retriever(self, query):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        vector_store = Chroma(
            persist_directory='./chroma_store',
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )
        retriever = vector_store.as_retriever()
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.configObj.GEMINI_API_KEY)
        # RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True  # Optional: show source context
        )
        return qa_chain(query)