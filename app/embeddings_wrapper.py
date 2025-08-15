# embeddings_wrapper.py
from typing import Literal, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
class EmbeddingProvider:
    def __init__(
        self,
        provider: Literal["gemini", "openai", "local", "ollama"],
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = model
        self.embedding_client = self._load_provider()

    def _load_provider(self):
        if self.provider == "gemini":
            return GoogleGenerativeAIEmbeddings(
                model=self.model or "models/embedding-001",
                google_api_key=self.api_key
            )
        elif self.provider == "ollama":
            return OllamaEmbeddings(
                model=self.model or "nomic-embed-text"
            )
        elif self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.model or "text-embedding-3-large",
                openai_api_key=self.api_key
            )
        elif self.provider == "local":
            return HuggingFaceEmbeddings(
                model_name=self.model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    def getEmbedingFunction(self):
        if self.provider == "gemini":
            return GoogleGenerativeAIEmbeddings(
                model=self.model or "models/embedding-001"
            )
        elif self.provider == "ollama":
            return OllamaEmbeddings(
                model=self.model or "nomic-embed-text"
            )
        elif self.provider == "openai":
            return OpenAIEmbeddings(
                model=self.model or "text-embedding-3-large"
            )
        elif self.provider == "local":
            return HuggingFaceEmbeddings(
                model_name=self.model or "sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")