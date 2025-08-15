from config import Config
from app_logger import AppLogger
import logging
import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_xai import ChatXAI
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from embeddings_wrapper import EmbeddingProvider
import docx
import asyncio

class RagPipline:
    fileContent = ""
    def __init__(self, llmModel):
        self.configObj = Config()
        self.loggerObj = AppLogger("rag_pipleline", logging.INFO).setupLogger()
        self.llmModel = llmModel or "gemini"
        
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
                self.fileContent += page.page_content
            self.loggerObj.info(f"file content {str(self.fileContent)}")
            self.embeddingsTexts()
    def loadDocx(self, docFile):
        folderPath = os.path.join(self.configObj.ROOT_PATH, 'data')
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        filePath = os.path.join(self.configObj.ROOT_PATH, 'data', 'temp.docx')
        self.loggerObj.info(f"file{filePath}")
        with open(filePath, "w", encoding="utf-8") as file:
            doc = docx.Document(docFile)
            text = "\n".join([para.text for para in doc.paragraphs])
            file.write(text)
            self.fileContent = text
            self.loggerObj.info(f"file content {str(self.fileContent)}")
            self.embeddingsTexts()
    def embeddingsTexts(self):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunk = text_splitter.split_text(self.fileContent)
        # embiddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        embiddings = EmbeddingProvider(
            provider=self.llmModel,
            api_key= self.configObj.LLM_CONFIG[self.llmModel]['api_key']
        ).embedding_client
        # embiddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        Chroma.from_texts(
            texts=chunk,
            embedding=embiddings,
            persist_directory='./chroma_store'
        )
        # res = vectore_store.persist()
    def retriever(self, query):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        try:
            llmConfig = self.configObj.LLM_CONFIG[self.llmModel]
            self.loggerObj.info(f"api key {llmConfig['api_key']}")
        except KeyError as e:
            self.loggerObj.error(f"Missing LLM config for {self.llmModel}: {e}")
            return {"error": str(e)}
        
        try:
            embiddingsFun = EmbeddingProvider(
                provider=self.llmModel,
                api_key= llmConfig['api_key']
            )
            vector_store = Chroma(
                persist_directory='./chroma_store',
                embedding_function=embiddingsFun.getEmbedingFunction()
            )
            retriever = vector_store.as_retriever()
        except Exception as e:
            self.loggerObj.error(f"Retriever setup failed: {e}")
            return {"error": str(e)}
        try:
            kwargs = {
                "model": llmConfig["model"],
                llmConfig["api_key_name"]: llmConfig["api_key"]
            }
            if self.llmModel == 'ollama':
                kwargs = {
                    "model": llmConfig["model"]
                }
            self.loggerObj.info(f"kwargs {kwargs}")
            llm = llmConfig["class"](**kwargs)
            # llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.configObj.GEMINI_API_KEY)
            # RAG Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True  # Optional: show source context
            )
            # result = qa_chain(query)
            result = qa_chain.invoke(query)
            if "source_documents" in result:
                for i, doc in enumerate(result["source_documents"], start=1):
                    self.loggerObj.info(f"Source {i}: {doc.metadata}")
                    self.loggerObj.info(f"Content: {doc.page_content[:200]}...")  # show first 200 chars
            else:
                self.loggerObj.warning("No source documents returned!")
            return result
        except Exception as e:
            self.loggerObj.info(f"result {str(e)}")
            # return {"result" : str(e)}
            return {"error": str(e)}