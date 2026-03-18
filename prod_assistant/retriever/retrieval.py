import os
from dotenv import load_dotenv
from langchain_astradb import AstraDBVectorStore

from prod_assistant.utils.config_loader import load_config
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
)


class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever_instance = None

    def _load_env_variables(self):
        load_dotenv()

        required_vars = [
            "GROQ_API_KEY",
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE",
        ]

        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv("GROQ_API_KEY")
        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def load_retriever(self):
        if not self.vstore:
            collection_name = self.config["astra_db"]["collection_name"]

            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )

        if not self.retriever_instance:
            top_k = self.config["retriever"]["top_k"] if "retriever" in self.config else 3

            self.retriever_instance = self.vstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k,
                    "fetch_k": 20,
                    "lambda_mult": 0.7,
                },
            )
            print("Retriever loaded successfully.")

        return self.retriever_instance

    def call_retriever(self, query):
        retriever = self.load_retriever()
        output = retriever.invoke(query)
        return output


def format_docs_for_eval(docs):
    if not docs:
        return []

    formatted_chunks = []

    for item in docs:
        d = item[0] if isinstance(item, tuple) else item
        meta = d.metadata or {}

        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)

    return formatted_chunks


def format_docs_for_display(docs) -> str:
    if not docs:
        return "No relevant documents found."

    formatted_chunks = []

    for item in docs:
        d = item[0] if isinstance(item, tuple) else item
        meta = d.metadata or {}

        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)

    return "\n\n---\n\n".join(formatted_chunks)


if __name__ == "__main__":
    user_query = "Can you suggest good budget iPhone?"

    retriever_obj = Retriever()
    retrieved_docs = retriever_obj.call_retriever(user_query)

    retrieved_contexts = format_docs_for_eval(retrieved_docs)
    display_context = format_docs_for_display(retrieved_docs)

    response = "budget iPhone"

    if not retrieved_contexts:
        print("No documents retrieved. Skipping evaluation.")
    else:
        context_score = evaluate_context_precision(user_query, response, retrieved_contexts)
        relevancy_score = evaluate_response_relevancy(user_query, response, retrieved_contexts)

        print("\n--- Evaluation Metrics ---")
        print("Context Precision Score:", context_score)
        print("Response Relevancy Score:", relevancy_score)

    print("\n--- Retrieved Context ---")
    print(display_context)

    for idx, doc in enumerate(retrieved_docs, 1):
        print(f"\nResult {idx}:")
        print(doc.page_content)
        print(f"Metadata: {doc.metadata}")