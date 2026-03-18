import asyncio
from typing import List
import grpc.experimental.aio as grpc_aio

from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy

from prod_assistant.utils.model_loader import ModelLoader

grpc_aio.init_grpc_aio()
model_loader = ModelLoader()


def evaluate_context_precision(query: str, response: str, retrieved_contexts: List[str]):
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )

        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)

            metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
            result = await metric.single_turn_ascore(sample)
            return result

        return asyncio.run(main())

    except Exception as e:
        return f"Evaluation failed: {e}"


def evaluate_response_relevancy(query: str, response: str, retrieved_contexts: List[str]):
    try:
        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )

        async def main():
            llm = model_loader.load_llm()
            evaluator_llm = LangchainLLMWrapper(llm)

            embedding_model = model_loader.load_embeddings()
            evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)

            metric = ResponseRelevancy(
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                strictness=1,
            )

            result = await metric.single_turn_ascore(sample)
            return result

        return asyncio.run(main())

    except Exception as e:
        return f"Evaluation failed: {e}"