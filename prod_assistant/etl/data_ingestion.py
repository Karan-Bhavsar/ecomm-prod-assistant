import os
import math
from typing import List, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore

from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config


class DataIngestion:
    """
    Handles CSV loading, cleaning, document creation, and ingestion into AstraDB.
    """

    def __init__(self):
        print("Initializing DataIngestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.config = load_config()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()

    def _load_env_variables(self):
        """
        Load and validate required Astra DB environment variables.
        """
        load_dotenv()

        required_vars = [
            "ASTRA_DB_API_ENDPOINT",
            "ASTRA_DB_APPLICATION_TOKEN",
            "ASTRA_DB_KEYSPACE",
        ]

        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.db_api_endpoint = os.getenv("ASTRA_DB_API_ENDPOINT")
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv("ASTRA_DB_KEYSPACE")

    def _get_csv_path(self) -> str:
        """
        Resolve CSV file path.
        """
        current_dir = os.getcwd()
        csv_path = os.path.join(current_dir, "data", "product_reviews.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self) -> pd.DataFrame:
        """
        Load CSV and validate expected columns.
        """
        df = pd.read_csv(self.csv_path)

        expected_columns = {
            "product_id",
            "product_title",
            "rating",
            "total_reviews",
            "price",
            "top_reviews",
        }

        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(f"CSV must contain columns: {expected_columns}")

        return df

    def _clean_value(self, value: Any) -> Any:
        """
        Convert invalid/missing values into safe Python values.
        """
        if pd.isna(value):
            return None

        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None

        if isinstance(value, str):
            cleaned = value.strip()

            if cleaned == "":
                return None

            if cleaned.lower() in {"n/a", "na", "none", "null", "nan"}:
                return None

            return cleaned

        return value

    def _safe_string(self, value: Any, default: str = "N/A") -> str:
        """
        Return a clean string for content generation.
        """
        cleaned = self._clean_value(value)
        return default if cleaned is None else str(cleaned)

    def _is_invalid_row(self, row: pd.Series) -> bool:
        """
        Decide whether a row should be skipped.
        """
        product_id = self._clean_value(row.get("product_id"))
        product_title = self._clean_value(row.get("product_title"))
        top_reviews = self._clean_value(row.get("top_reviews"))

        if product_id is None:
            return True

        if product_title is None:
            return True

        if top_reviews is None:
            return True

        if str(top_reviews).strip().lower() == "invalid product url":
            return True

        return False

    def transform_data(self) -> List[Document]:
        """
        Convert CSV rows into LangChain Documents.
        Important product fields are included in page_content so vector search
        can retrieve them semantically.
        """
        documents: List[Document] = []
        skipped_rows = 0

        for _, row in self.product_data.iterrows():
            if self._is_invalid_row(row):
                skipped_rows += 1
                continue

            product_id = self._clean_value(row["product_id"])
            product_title = self._clean_value(row["product_title"])
            rating = self._clean_value(row["rating"])
            total_reviews = self._clean_value(row["total_reviews"])
            price = self._clean_value(row["price"])
            top_reviews = self._clean_value(row["top_reviews"])

            metadata = {
                "product_id": product_id,
                "product_title": product_title,
                "rating": rating,
                "total_reviews": total_reviews,
                "price": price,
            }

            content = (
                f"Product Title: {self._safe_string(product_title)}\n"
                f"Price: {self._safe_string(price)}\n"
                f"Rating: {self._safe_string(rating)}\n"
                f"Total Reviews: {self._safe_string(total_reviews)}\n"
                f"Top Reviews: {self._safe_string(top_reviews, default='No review available')}"
            )

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        print(f"Transformed {len(documents)} documents.")
        print(f"Skipped {skipped_rows} invalid rows.")
        return documents

    def store_in_vector_db(self, documents: List[Document]):
        """
        Store transformed documents in AstraDB vector store.
        """
        collection_name = self.config["astra_db"]["collection_name"]

        embedding_model = self.model_loader.load_embeddings()

        vstore = AstraDBVectorStore(
            embedding=embedding_model,
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vstore.add_documents(documents)
        print(f"Successfully inserted {len(inserted_ids)} documents into AstraDB.")
        return vstore, inserted_ids

    def run_pipeline(self):
        """
        Run full ingestion pipeline and test with a sample query.
        """
        documents = self.transform_data()

        if not documents:
            raise ValueError("No valid documents available for ingestion.")

        vstore, _ = self.store_in_vector_db(documents)

        query = "Can you tell me the low budget iphone?"
        results = vstore.similarity_search(query, k=3)

        print(f"\nSample search results for query: '{query}'")
        for i, res in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Content:\n{res.page_content}")
            print(f"Metadata:\n{res.metadata}")

    def preview_documents(self, n: int = 5):
        """
        Preview transformed documents without storing them.
        Useful for debugging.
        """
        documents = self.transform_data()
        print(f"\nShowing first {min(n, len(documents))} transformed documents:\n")

        for i, doc in enumerate(documents[:n], start=1):
            print(f"Document {i}")
            print("Content:")
            print(doc.page_content)
            print("Metadata:")
            print(doc.metadata)
            print("-" * 80)


if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.run_pipeline()