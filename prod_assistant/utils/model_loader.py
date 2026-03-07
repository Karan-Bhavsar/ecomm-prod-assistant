import os
import sys
from typing import Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from prod_assistant.utils.config_loader import load_config
from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


class LocalSentenceTransformerEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings

        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            raise ProductAssistantException(
                f"Failed to initialize SentenceTransformer model: {model_name}",
                sys,
            ) from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self.model.encode(
                texts,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        except Exception as e:
            raise ProductAssistantException("Failed to embed documents", sys) from e

    def embed_query(self, text: str) -> list[float]:
        try:
            vector = self.model.encode(
                text,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return vector.tolist()
        except Exception as e:
            raise ProductAssistantException("Failed to embed query", sys) from e


class ApiKeyManager:
    def __init__(self):
        load_dotenv()

        self.api_keys = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY"),
            "ASTRA_DB_API_ENDPOINT": os.getenv("ASTRA_DB_API_ENDPOINT"),
            "ASTRA_DB_APPLICATION_TOKEN": os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            "ASTRA_DB_KEYSPACE": os.getenv("ASTRA_DB_KEYSPACE"),
        }

        for key, val in self.api_keys.items():
            if val:
                log.info(f"{key} loaded from environment")
            else:
                log.warning(f"{key} is missing from environment")

    def get(self, key: str) -> Optional[str]:
        return self.api_keys.get(key)


class ModelLoader:
    def __init__(self):
        load_dotenv()
        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()
        log.info("YAML config loaded", config_keys=list(self.config.keys()))

    def _normalize_provider(self, provider: Optional[str]) -> Optional[str]:
        if not provider:
            return None
        return provider.strip().lower()

    def _has_required_key(self, provider: str) -> bool:
        if provider == "groq":
            return bool(self.api_key_mgr.get("GROQ_API_KEY"))
        if provider == "google":
            return bool(self.api_key_mgr.get("GOOGLE_API_KEY"))
        if provider == "openai":
            return bool(self.api_key_mgr.get("OPENAI_API_KEY"))
        return False

    def _resolve_llm_provider(self) -> str:
        """
        Resolution priority:
        1. LLM_PROVIDER env var
        2. llm.default_provider in config.yaml (optional)
        3. First available provider with valid key in order: groq -> google -> openai
        """
        llm_block = self.config.get("llm", {})

        env_provider = self._normalize_provider(os.getenv("LLM_PROVIDER"))
        config_default = self._normalize_provider(llm_block.get("default_provider"))

        candidate_order = []

        if env_provider:
            candidate_order.append(env_provider)

        if config_default and config_default not in candidate_order:
            candidate_order.append(config_default)

        for fallback in ["groq", "google", "openai"]:
            if fallback not in candidate_order:
                candidate_order.append(fallback)

        for provider in candidate_order:
            if provider not in llm_block:
                continue
            if self._has_required_key(provider):
                log.info("Resolved LLM provider", provider=provider)
                return provider

        raise ValueError(
            "No usable LLM provider found. Set LLM_PROVIDER in .env or add a valid API key "
            "for one of: groq, google, openai."
        )

    def load_embeddings(self):
        try:
            embedding_config = self.config["embedding_model"]
            provider = embedding_config.get("provider", "local")
            model_name = embedding_config["model_name"]
            normalize = embedding_config.get("normalize", True)
            device = os.getenv("EMBEDDINGS_DEVICE", "cpu")

            provider = self._normalize_provider(provider)

            if provider != "local":
                raise ValueError(
                    f"Embedding provider '{provider}' is not supported in local mode. "
                    f"Set provider='local' and use a SentenceTransformer model."
                )

            log.info(
                "Loading LOCAL embedding model",
                model=model_name,
                device=device,
                normalize=normalize,
            )

            return LocalSentenceTransformerEmbeddings(
                model_name=model_name,
                device=device,
                normalize_embeddings=normalize,
            )

        except Exception as e:
            log.error("Error loading local embedding model", error=str(e))
            raise ProductAssistantException(
                "Failed to load local embedding model", sys
            ) from e

    def load_llm(self):
        try:
            llm_block = self.config["llm"]
            provider_key = self._resolve_llm_provider()

            if provider_key not in llm_block:
                raise ValueError(
                    f"LLM provider '{provider_key}' not found in config under 'llm'"
                )

            llm_config = llm_block[provider_key]
            provider = self._normalize_provider(llm_config.get("provider", provider_key))
            model_name = llm_config.get("model_name")
            temperature = llm_config.get("temperature", 0.2)
            max_tokens = llm_config.get("max_output_tokens", 2048)

            if not model_name:
                raise ValueError(f"Missing model_name for provider '{provider_key}'")

            log.info("Loading LLM", provider=provider, model=model_name)

            if provider == "google":
                api_key = self.api_key_mgr.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY is missing")
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            if provider == "groq":
                api_key = self.api_key_mgr.get("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY is missing")
                return ChatGroq(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature,
                )

            if provider == "openai":
                api_key = self.api_key_mgr.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is missing. If you do not want OpenAI, set LLM_PROVIDER=groq or LLM_PROVIDER=google."
                    )
                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            raise ValueError(f"Unsupported LLM provider: {provider}")

        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise

if __name__ == "__main__":
    loader = ModelLoader()

    embeddings = loader.load_embeddings()
    print(f"Embedding Model Loaded: {embeddings}")

    vec = embeddings.embed_query("Hello, how are you?")
    print(f"Embedding Vector Length: {len(vec)}")
    print(f"First 10 values: {vec[:10]}")

    llm = loader.load_llm()
    print(f"LLM Loaded: {llm}")