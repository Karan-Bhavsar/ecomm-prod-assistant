from typing import TypedDict, Literal, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
)


class AgenticRAG:
    """Simplified Agentic RAG pipeline with controlled retries and evaluation."""

    class AgentState(TypedDict):
        question: str
        current_query: str
        context: str
        retrieved_contexts: List[str]
        answer: str
        rewrite_count: int
        evaluation: str

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Helpers ----------
    def _format_docs(self, docs) -> str:
        if not docs:
            return "No relevant documents found."

        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Total Reviews: {meta.get('total_reviews', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    def _format_docs_for_eval(self, docs) -> List[str]:
        if not docs:
            return []

        formatted_chunks = []
        for d in docs:
            meta = d.metadata or {}
            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Total Reviews: {meta.get('total_reviews', 'N/A')}\n"
                f"Reviews:\n{d.page_content.strip()}"
            )
            formatted_chunks.append(formatted)

        return formatted_chunks

    # ---------- Nodes ----------
    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER ---")
        query = state["current_query"]

        retriever = self.retriever_obj.load_retriever()
        docs = retriever.invoke(query)

        context = self._format_docs(docs)
        retrieved_contexts = self._format_docs_for_eval(docs)

        return {
            "context": context,
            "retrieved_contexts": retrieved_contexts,
        }

    def _route_after_retrieval(self, state: AgentState) -> Literal["generator", "rewriter", "end"]:
        print("--- ROUTER ---")
        context = state["context"]

        if context and context.strip() != "No relevant documents found.":
            return "generator"

        if state["rewrite_count"] < 1:
            return "rewriter"

        return "end"

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["question"]

        prompt = ChatPromptTemplate.from_template(
            """Rewrite the following user query to improve retrieval for an ecommerce product assistant.
Keep it short, clear, and product-focused.

Original query: {question}

Rewritten query:"""
        )

        chain = prompt | self.llm | StrOutputParser()
        new_query = chain.invoke({"question": question}).strip()

        return {
            "current_query": new_query,
            "rewrite_count": state["rewrite_count"] + 1,
        }

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke(
            {
                "context": state["context"],
                "question": state["question"],
            }
        )

        evaluation_summary = "Evaluation skipped."

        try:
            if state["retrieved_contexts"]:
                context_score = evaluate_context_precision(
                    state["question"],
                    response,
                    state["retrieved_contexts"],
                )

                relevancy_score = evaluate_response_relevancy(
                    state["question"],
                    response,
                    state["retrieved_contexts"],
                )

                evaluation_summary = (
                    f"Context Precision Score: {context_score}\n"
                    f"Response Relevancy Score: {relevancy_score}"
                )
            else:
                evaluation_summary = "No retrieved contexts available for evaluation."

        except Exception as e:
            evaluation_summary = f"Evaluation failed: {e}"

        return {
            "answer": response,
            "evaluation": evaluation_summary,
        }

    def _fallback(self, state: AgentState):
        print("--- FALLBACK ---")
        return {
            "answer": "I could not find relevant product information in the database for your question.",
            "evaluation": "Evaluation skipped because no relevant context was found.",
        }

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)

        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Fallback", self._fallback)

        workflow.add_edge(START, "Retriever")

        workflow.add_conditional_edges(
            "Retriever",
            self._route_after_retrieval,
            {
                "generator": "Generator",
                "rewriter": "Rewriter",
                "end": "Fallback",
            },
        )

        workflow.add_edge("Rewriter", "Retriever")
        workflow.add_edge("Generator", END)
        workflow.add_edge("Fallback", END)

        return workflow

    # ---------- Public Run ----------
    def run(self, query: str, thread_id: str = "default_thread") -> dict:
        result = self.app.invoke(
            {
                "question": query,
                "current_query": query,
                "context": "",
                "retrieved_contexts": [],
                "answer": "",
                "rewrite_count": 0,
                "evaluation": "",
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        return {
            "answer": result["answer"],
            "evaluation": result["evaluation"],
            "context": result["context"],
        }


if __name__ == "__main__":
    rag_agent = AgenticRAG()
    result = rag_agent.run("Compare iPhone 15 vs iPhone 16 based on customer reviews, price, and overall value. Which one should I buy?")

    print("\nFinal Answer:\n", result["answer"])
    print("\nEvaluation:\n", result["evaluation"])
    print("\nRetrieved Context:\n", result["context"])