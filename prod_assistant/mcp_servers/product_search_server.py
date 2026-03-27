from mcp.server.fastmcp import FastMCP
from prod_assistant.retriever.retrieval import Retriever
from ddgs import DDGS
import re

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Load retriever once
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# ---------- Helpers ----------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_query(query: str) -> list[str]:
    stopwords = {
        "the", "a", "an", "for", "of", "with", "and", "or", "to", "in",
        "on", "price", "buy", "best", "latest", "phone", "mobile"
    }
    tokens = normalize_text(query).split()
    return [t for t in tokens if t not in stopwords and len(t) > 1]

def doc_text(doc) -> str:
    meta = doc.metadata or {}
    combined = " ".join([
        str(meta.get("product_title", "")),
        str(meta.get("price", "")),
        str(meta.get("rating", "")),
        str(doc.page_content or "")
    ])
    return normalize_text(combined)

def is_relevant_result(query: str, docs, min_matches: int = 2) -> bool:
    if not docs:
        return False

    q_tokens = tokenize_query(query)
    if not q_tokens:
        return True

    all_doc_text = " ".join(doc_text(d) for d in docs)
    matched_tokens = [token for token in q_tokens if token in all_doc_text]

    print(f"[DEBUG] Query tokens: {q_tokens}")
    print(f"[DEBUG] Matched tokens: {matched_tokens}")

    return len(matched_tokens) >= min_matches

def format_docs(docs) -> str:
    if not docs:
        return ""

    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{(d.page_content or '').strip()}"
        )
        formatted_chunks.append(formatted)

    return "\n\n---\n\n".join(formatted_chunks)

def ddg_search(query: str, max_results: int = 5) -> str:
    """
    Search using DDGS directly instead of LangChain wrapper.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "NO_WEB_RESULTS"

        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "N/A")
            body = r.get("body", "N/A")
            href = r.get("href", "N/A")

            formatted.append(
                f"{i}. {title}\n"
                f"Snippet: {body}\n"
                f"URL: {href}"
            )

        return "\n\n".join(formatted)

    except Exception as e:
        return f"ERROR_WEB_SEARCH: {str(e)}"

# ---------- MCP Tools ----------
@mcp.tool()
async def get_product_info(query: str) -> str:
    """Retrieve product information for a given query from local retriever."""
    try:
        docs = retriever.invoke(query)

        if not docs:
            return "NO_LOCAL_RESULTS"

        if not is_relevant_result(query, docs, min_matches=2):
            return "NO_LOCAL_RESULTS"

        context = format_docs(docs)
        if not context.strip():
            return "NO_LOCAL_RESULTS"

        return context

    except Exception as e:
        return f"ERROR_LOCAL_RETRIEVAL: {str(e)}"

@mcp.tool()
async def web_search(query: str) -> str:
    """Search the web using DDGS directly."""
    return ddg_search(query)

# ---------- Run Server ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")