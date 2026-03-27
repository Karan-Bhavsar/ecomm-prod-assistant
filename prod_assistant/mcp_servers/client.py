import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    client = MultiServerMCPClient({
        "hybrid_search": {
            "command": "python",
            "args": [
                r"C:\Users\Karan\ecomm-prod-assistant\prod_assistant\mcp_servers\product_search_server.py"
            ],
            "transport": "stdio",
        }
    })

    tools = await client.get_tools()
    print("Available tools:", [t.name for t in tools])

    retriever_tool = next(t for t in tools if t.name == "get_product_info")
    web_tool = next(t for t in tools if t.name == "web_search")

    query = "Samsung Galaxy S25 price"

    retriever_result = await retriever_tool.ainvoke({"query": query})
    print("\nRetriever Result:\n", retriever_result)

    if (
        not retriever_result
        or not retriever_result.strip()
        or "NO_LOCAL_RESULTS" in retriever_result
        or "ERROR_LOCAL_RETRIEVAL" in retriever_result
    ):
        print("\nNo relevant local results, falling back to web search...\n")
        web_result = await web_tool.ainvoke({"query": query})
        print("Web Search Result:\n", web_result)

if __name__ == "__main__":
    asyncio.run(main())