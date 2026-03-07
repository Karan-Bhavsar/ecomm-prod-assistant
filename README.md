uv pip install -e. --no-deps
uv pip install -r requirments.txt
uv run python -m prod_assistant.retriever.retrieval.py
streamlit run scrapper_ui.py