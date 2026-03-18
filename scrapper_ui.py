import streamlit as st
from prod_assistant.etl.data_scrapper import FlipkartScraper
from prod_assistant.etl.data_ingestion import DataIngestion
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Shop Buddy AI",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
    <style>
        .main {
            padding-top: 1.5rem;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .hero-box {
            background: linear-gradient(135deg, #1f2937, #111827);
            padding: 1.8rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .section-box {
            background-color: #111827;
            padding: 1.2rem;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1rem;
        }
        .small-note {
            color: #9ca3af;
            font-size: 0.92rem;
        }
        .metric-box {
            background-color: #0f172a;
            padding: 1rem;
            border-radius: 14px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .stButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }
        .stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
            font-weight: 600;
            padding: 0.65rem 1rem;
        }
    </style>
""", unsafe_allow_html=True)

flipkart_scraper = FlipkartScraper()
output_path = "data/product_reviews.csv"

# -----------------------------
# Session State
# -----------------------------
if "product_inputs" not in st.session_state:
    st.session_state.product_inputs = [""]

def add_product_input():
    st.session_state.product_inputs.append("")

# -----------------------------
# Hero Section
# -----------------------------
st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom:0.4rem;">🛍️ Shop Buddy AI</h1>
    <p style="font-size:1.05rem; margin-bottom:0.2rem;">
        Flipkart Product Review Scraper + AstraDB Vector Ingestion
    </p>
    <p class="small-note">
        Search products, collect reviews, save them as CSV, and push the scraped data into your vector database.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([2.1, 1], gap="large")

with left_col:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📝 Product Search Setup")

    product_description = st.text_area(
        "Optional Product Description",
        placeholder="Example: budget gaming laptop, lightweight sneakers, wireless earbuds under 3000...",
        help="This description is used as an additional search keyword."
    )

    st.markdown("#### 🛒 Product Names")
    updated_inputs = []
    for i, val in enumerate(st.session_state.product_inputs):
        input_val = st.text_input(
            f"Product {i+1}",
            value=val,
            key=f"product_{i}",
            placeholder=f"Enter product name {i+1}"
        )
        updated_inputs.append(input_val)
    st.session_state.product_inputs = updated_inputs

    add_col1, add_col2 = st.columns([1, 3])
    with add_col1:
        st.button("➕ Add Another Product", on_click=add_product_input)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("⚙️ Scraping Controls")

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        max_products = st.number_input(
            "How many products per search?",
            min_value=1,
            max_value=10,
            value=1
        )
    with control_col2:
        review_count = st.number_input(
            "How many reviews per product?",
            min_value=1,
            max_value=10,
            value=2
        )

    start_scrape = st.button("🚀 Start Scraping")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("📊 Current Selection")

    filled_products = [p.strip() for p in st.session_state.product_inputs if p.strip()]

    st.markdown(f"""
    <div class="metric-box">
        <h3 style="margin:0;">{len(filled_products)}</h3>
        <p style="margin:0;">Product Queries</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <h3 style="margin:0;">{max_products}</h3>
        <p style="margin:0;">Products per Search</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <h3 style="margin:0;">{review_count}</h3>
        <p style="margin:0;">Reviews per Product</p>
    </div>
    """, unsafe_allow_html=True)

    if product_description.strip():
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.info(f"**Description Keyword:** {product_description.strip()}")

    if filled_products:
        st.markdown("#### Entered Products")
        for item in filled_products:
            st.markdown(f"- {item}")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Scraping Logic
# -----------------------------
if start_scrape:
    product_inputs = [p.strip() for p in st.session_state.product_inputs if p.strip()]
    if product_description.strip():
        product_inputs.append(product_description.strip())

    if not product_inputs:
        st.warning("⚠️ Please enter at least one product name or a product description.")
    else:
        final_data = []

        progress_bar = st.progress(0)
        status_placeholder = st.empty()

        for idx, query in enumerate(product_inputs):
            status_placeholder.info(f"🔍 Searching for: {query}")
            results = flipkart_scraper.scrape_flipkart_products(
                query,
                max_products=max_products,
                review_count=review_count
            )
            final_data.extend(results)
            progress_bar.progress((idx + 1) / len(product_inputs))

        unique_products = {}
        for row in final_data:
            if row[1] not in unique_products:
                unique_products[row[1]] = row

        final_data = list(unique_products.values())
        st.session_state["scraped_data"] = final_data
        flipkart_scraper.save_to_csv(final_data, output_path)

        status_placeholder.success("✅ Scraping completed successfully!")
        st.success("✅ Data saved to `data/product_reviews.csv`")

        result_col1, result_col2 = st.columns([1, 1])
        with result_col1:
            st.metric("Total Unique Products Scraped", len(final_data))
        with result_col2:
            st.metric("Search Queries Used", len(product_inputs))

        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download CSV",
                    data=f,
                    file_name="product_reviews.csv",
                    mime="text/csv"
                )

# -----------------------------
# Vector DB Ingestion
# -----------------------------
st.markdown("<br>", unsafe_allow_html=True)

if "scraped_data" in st.session_state:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("🧠 Vector Database Storage")
    st.write("Push the scraped review data into **AstraDB** for semantic retrieval and downstream AI workflows.")

    if st.button("📡 Store in Vector DB (AstraDB)"):
        with st.spinner("Initializing ingestion pipeline..."):
            try:
                ingestion = DataIngestion()
                st.info("🚀 Running ingestion pipeline...")
                ingestion.run_pipeline()
                st.success("✅ Data successfully ingested to AstraDB!")
            except Exception as e:
                st.error("❌ Ingestion failed!")
                st.exception(e)

    st.markdown('</div>', unsafe_allow_html=True)