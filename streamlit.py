import streamlit as st
import pandas as pd
import random
from PIL import Image, ImageOps
import requests
from io import BytesIO
from recommendation_system.recommendation_system import build_models, get_hybrid_recommendations
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from feature_extraction import enrich_review_features
import os 
import joblib


# ---------- Page Config ----------
st.set_page_config(layout="wide")
st.header("🧠 Multimodal Recommender System Demo")

@st.cache_resource
def load_models(df):
    embedding_cols = [
        "item_desc_features",
        "item_desc_keyword_features",
        "image_features",
        "review_features",
        "review_keyword_features"
    ]

    for col in embedding_cols:
        df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    scaler = MinMaxScaler()
    columns_to_normalize = ['average_rating','sentiment_score','weighted_rating', 'rating_number', 'emotion_score']
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return (*build_models(df), scaler)

df = joblib.load("df.pkl")
best_cb_model, best_cf_model, best_mm_model, scaler = load_models(df)

# ---------- Utility ----------
def load_and_pad_image(url, size=(224, 224), padding_color=(0, 0, 0)):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = ImageOps.pad(img, size, method=Image.BICUBIC, color=padding_color, centering=(0.5, 0.5))
    return img

def truncate_title(title, max_length=50):
    return title if len(title) <= max_length else title[:max_length] + "..."

# ---------- User Setup ----------
user_ids = df['user_id'].unique().tolist()
if 'random_user_ids' not in st.session_state:
    st.session_state.random_user_ids = random.sample(user_ids, 4)
if 'user_dataset' not in st.session_state:
    st.session_state.user_dataset = pd.DataFrame()
if 'last_selected_user' not in st.session_state:
    st.session_state.last_selected_user = None

st.sidebar.markdown("#### 👤 Select a User")
with st.sidebar.container(border=True):
    options = ['New User'] + st.session_state.random_user_ids
    selected_user = st.radio("User Options", options, label_visibility="collapsed")

# Only update dataset when user changes
if selected_user != st.session_state.last_selected_user:
    if selected_user == 'New User':
        st.session_state.user_dataset = pd.DataFrame(columns=df.columns)
    else:
        user_data = df[df['user_id'] == selected_user].copy()

        # 🔧 Convert embedded features to np.array
        for col in ["review_features", "review_keyword_features"]:
            user_data[col] = user_data[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

        st.session_state.user_dataset = user_data

    st.session_state.last_selected_user = selected_user

user_dataset = st.session_state.user_dataset

# ---------- Sidebar Purchases ----------
if not user_dataset.empty:
    st.sidebar.markdown("#### 🛍️ Past Purchases")
    with st.sidebar.container():
        for _, row in user_dataset.head(5).iterrows():
            with st.sidebar.container(border=True):
                cols = st.columns([1, 2])
                with cols[0]:
                    st.image(load_and_pad_image(row['large_image_link']), width=60)
                with cols[1]:
                    st.markdown(f"**Rating:** ⭐ {row['rating']}")
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"{row['text']}")
else:
    st.sidebar.info("No past purchases for new users.")

# ---------- Search Input ----------
if "search_triggered" not in st.session_state:
    st.session_state.search_triggered = False
if "last_search" not in st.session_state:
    st.session_state.last_search = ""
if "reset_search" not in st.session_state:
    st.session_state.reset_search = False

if st.session_state.reset_search:
    st.session_state["search_input"] = ""
    st.session_state["last_search"] = ""
    st.session_state["search_triggered"] = False
    st.session_state["reset_search"] = False
    st.rerun()

search_input = st.text_input("Search for a product...", key="search_input")

if search_input and search_input != st.session_state.last_search and not st.session_state.search_triggered:
    st.session_state.search_triggered = True
    st.session_state.last_search = search_input
    st.query_params["page"] = "search"
    st.query_params["q"] = search_input
    st.query_params.pop("asin", None)
    st.rerun()
# ---------- Weights Input ---------
col1, col2, col3 = st.columns(3)

# Determine if user is new
is_new_user = user_dataset.empty

# Set default weights depending on user type
default_cf = 0.0 if is_new_user else 0.3
default_cb = 0.5 if is_new_user else 0.3
default_mm = 0.5 if is_new_user else 0.4

# Load defaults into session state if not set
if "weight_cf" not in st.session_state:
    st.session_state.weight_cf = default_cf
if "weight_cb" not in st.session_state:
    st.session_state.weight_cb = default_cb
if "weight_mm" not in st.session_state:
    st.session_state.weight_mm = default_mm

# UI: numeric inputs (can be sliders if you prefer)
with col1:
    cf_input = st.number_input(
        "Collaborative Weights",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weight_cf,
        step=0.01,
        disabled=is_new_user,
        key="cf_weight_input"
    )
with col2:
    cb_input = st.number_input(
        "Content Based Weights",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weight_cb,
        step=0.01,
        key="cb_weight_input"
    )
with col3:
    mm_input = st.number_input(
        "Multimodal Weights",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.weight_mm,
        step=0.01,
        key="mm_weight_input"
    )

# Auto-adjust weights to sum to 1
if is_new_user:
    cf_input = 0.0
    total = cb_input + mm_input
    cb_input = round(cb_input / total, 2)
    mm_input = round(mm_input / total, 2)
else:
    total = cf_input + cb_input + mm_input
    cf_input = round(cf_input / total, 2)
    cb_input = round(cb_input / total, 2)
    mm_input = round(mm_input / total, 2)

# Save normalized values back to session state
st.session_state.weight_cf = cf_input
st.session_state.weight_cb = cb_input
st.session_state.weight_mm = mm_input
    
# ---------- Query Params ----------
query = st.query_params
page = query.get("page", "home")
selected_asin = query.get("asin", None)
search_query = query.get("q", "")
from_search = query.get("from_search", "false") == "true"

# ---------- Home Page ----------
def show_home():
    if not user_dataset.empty:
        st.subheader("🤖 Top 10 Personalised Recommendations")

        top_recs = get_hybrid_recommendations(
            input_features=user_dataset.iloc[[-1]],
            dataset=df,
            best_cf_model=best_cf_model,
            best_cb_model=best_cb_model,
            best_mm_model=best_mm_model,
            top_n=10,
            weight_cf=st.session_state.weight_cf if not user_dataset.empty else 0.0,
            weight_cb=st.session_state.weight_cb,
            weight_mm=st.session_state.weight_mm
        )

        rec_items_df = df[df['parent_asin'].isin([asin for asin, _ in top_recs])].drop_duplicates('parent_asin')

        rows = [rec_items_df.iloc[i:i+5] for i in range(0, len(rec_items_df), 5)]
        for row_items in rows:
            cols = st.columns(len(row_items))
            for idx, (_, item) in enumerate(row_items.iterrows()):
                with cols[idx]:
                    with st.container(border=True, height=400):
                        img = load_and_pad_image(item['large_image_link'])
                        st.image(img)
                        st.caption(f"`{truncate_title(item['product_title'])}`")
                        if st.button("View", key=f"rec_{item['parent_asin']}"):
                            st.session_state["search_triggered"] = False
                            st.query_params.clear()
                            st.query_params["page"] = "detail"
                            st.query_params["asin"] = item['parent_asin']
                            st.query_params["from_search"] = "false"
                            st.rerun()

    # Always show top 10 most sold items
    st.subheader("🛒 Top 10 Most Sold Items")
    top_items = (
        df.groupby(['parent_asin', 'large_image_link', 'product_title'])
        .size()
        .reset_index(name='purchase_count')
        .sort_values(by='purchase_count', ascending=False)
        .head(10)
    )
    rows = [top_items.iloc[i:i+5] for i in range(0, 10, 5)]
    for row_items in rows:
        cols = st.columns(5)
        for idx, (_, item) in enumerate(row_items.iterrows()):
            with cols[idx]:
                with st.container(border=True, height=400):
                    img = load_and_pad_image(item['large_image_link'])
                    st.image(img)
                    st.caption(f"`{truncate_title(item['product_title'])}`")
                    if st.button("View", key=f"home_{item['parent_asin']}"):
                        st.session_state["search_triggered"] = False
                        st.query_params.clear()
                        st.query_params["page"] = "detail"
                        st.query_params["asin"] = item['parent_asin']
                        st.query_params["from_search"] = "false"
                        st.rerun()

    st.subheader("📄 Top 5 Rows of the Full Data")
    st.dataframe(df.head())


# ---------- Product Detail Page ----------
def apply_recommendation_fallbacks(df, base_df):
    df = df.copy()

    # Fill missing values
    df["weighted_rating"] = df["weighted_rating"].fillna(base_df["weighted_rating"].mean())
    df["sentiment_score"] = df["sentiment_score"].fillna(0)
    df["emotion_score"] = df["emotion_score"].fillna(0)

    df["review_features"] = df["review_features"].apply(
        lambda x: x if isinstance(x, np.ndarray) else np.zeros(512)
    )
    df["review_keyword_features"] = df["review_keyword_features"].apply(
        lambda x: x if isinstance(x, np.ndarray) else np.zeros(512)
    )

    # Ensure embedding columns are NumPy arrays
    embedding_cols = [
        "item_desc_features", "item_desc_keyword_features",
        "image_features", "review_features", "review_keyword_features"
    ]
    for col in embedding_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    return df

def show_product_detail(asin):
    product = df[df['parent_asin'] == asin].iloc[0]
    title = product.get("product_title", asin)
    st.subheader(f"🧾 Product Detail — {title}")
    st.image(load_and_pad_image(product["large_image_link"]), width=300)

    average_rating = product.get('average_rating', None)
    rating_number = product.get('rating_number', None)

    if average_rating is not None and rating_number is not None:
        columns_to_normalize = ['average_rating', 'sentiment_score', 'weighted_rating', 'rating_number', 'emotion_score']

        # Use actual values from the product row (already normalized)
        temp_row = product[columns_to_normalize].values.reshape(1, -1)
        unnormalized = scaler.inverse_transform(temp_row)[0]

        st.markdown(f"**Average Rating:** {round(unnormalized[0], 2)}")
        st.markdown(f"**Rating Count:** {int(unnormalized[3])}")
    else:
        st.markdown("**Average Rating:** N/A")
        st.markdown("**Rating Count:** N/A")

    for label, key in [
        ("📝 Features", "product_features"),
        ("🛍️ Description", "product_description"),
        ("🏬 Store", "product_store"),
        ("📦 Details", "product_details")
    ]:
        with st.expander(label):
            value = product.get(key, None)

            if isinstance(value, list) and value:
                for item in value:
                    st.markdown(f"- {item}")

            elif isinstance(value, dict) and value:
                for k, v in value.items():
                    st.markdown(f"- **{k}**: {v}")

            elif isinstance(value, str) and value.strip():
                st.markdown(f"- {value}")

            else:
                st.markdown("_No information available._")

    already_bought = (
        not user_dataset.empty and
        asin in user_dataset['parent_asin'].values
    )

    buy_label = "🛒 Buy This Again?" if already_bought else "🛒 Buy This Item"
    if st.button(buy_label):
        st.session_state["buy_clicked"] = True

    # Show review form after clicking buy
    if st.session_state.get("buy_clicked"):
        with st.form("review_form", clear_on_submit=True):
            rating = st.slider("Rate this product:", 1, 5, 5)
            title = st.text_input("Review Title")
            body = st.text_area("Your Review")
            submit = st.form_submit_button("Submit Review")

            if submit:
                review_text = f"{title} {body}".strip()
                enriched = enrich_review_features(review_text, rating)

                new_row = product.copy()
                new_row["rating"] = rating
                new_row["weighted_rating"] = rating

                # User ID logic
                if selected_user != "New User":
                    new_row["user_id"] = selected_user
                    user_row = df[df["user_id"] == selected_user].iloc[0]
                    new_row["user_id_encoded"] = user_row["user_id_encoded"]
                else:
                    new_row["user_id"] = "0"
                    new_row["user_id_encoded"] = 0

                # Add enriched features
                new_row.update(enriched)

                # Append and fallback-clean
                st.session_state.user_dataset = pd.concat(
                    [st.session_state.user_dataset, pd.DataFrame([new_row])],
                    ignore_index=True
                )
                st.session_state.user_dataset = apply_recommendation_fallbacks(st.session_state.user_dataset, df)

                st.success("✅ Added to your purchases with review!")
                st.session_state["buy_clicked"] = False
                st.rerun()

    if from_search:
        if st.button("🔙 Back to Search Results"):
            st.query_params["page"] = "search"
            st.query_params.pop("asin", None)
            st.query_params["from_search"] = "true"
            st.rerun()
        if st.button("🔙 Back to Main Page"):
            st.session_state["reset_search"] = True
            st.query_params.clear()
            st.rerun()
    else:
        if st.button("🔙 Back to Main Page"):
            st.session_state["reset_search"] = True
            st.query_params.clear()
            st.rerun()

    with st.expander("💬 Past Reviews"):
        reviews = df[df['parent_asin'] == asin][['user_id', 'rating', 'title','text']]
        if not reviews.empty:
            for _, row in reviews.iterrows():
                with st.container(border=True):
                    st.markdown(f"**👤 {row['user_id']}** — ⭐ {row['rating']}")
                    st.markdown(f"**{row['title']}**")
                    st.markdown(row['text'])
        else:
            st.info("No reviews yet.")
            
    st.subheader("🔁 You May Also Like")

    # Use real dataset if already bought, else simulate it
    base_input = user_dataset.copy()
    if asin not in base_input['parent_asin'].values:
        simulated_row = product.copy()
        base_input = pd.concat([base_input, pd.DataFrame([simulated_row])], ignore_index=True)

    # Apply fallbacks
    input_for_recs = apply_recommendation_fallbacks(base_input, df)
    
    with st.spinner("Generating recommendations..."):
        if not input_for_recs.empty:
            top_recs = get_hybrid_recommendations(
                input_features=input_for_recs.iloc[[-1]],
                dataset=df,
                best_cf_model=best_cf_model,
                best_cb_model=best_cb_model,
                best_mm_model=best_mm_model,
                top_n=10,
                weight_cf=st.session_state.weight_cf,
                weight_cb=st.session_state.weight_cb,
                weight_mm=st.session_state.weight_mm
            )

            rec_items_df = df[df['parent_asin'].isin([asin for asin, _ in top_recs])].drop_duplicates('parent_asin')

            rows = [rec_items_df.iloc[i:i + 5] for i in range(0, len(rec_items_df), 5)]
            for row_items in rows:
                cols = st.columns(len(row_items))
                for idx, (_, item) in enumerate(row_items.iterrows()):
                    with cols[idx]:
                        with st.container(border=True, height=400):
                            img = load_and_pad_image(item['large_image_link'])
                            st.image(img)
                            st.caption(f"`{truncate_title(item['product_title'])}`")
                            if st.button("View", key=f"rec_detail_{item['parent_asin']}"):
                                st.session_state["search_triggered"] = False
                                st.query_params.clear()
                                st.query_params["page"] = "detail"
                                st.query_params["asin"] = item["parent_asin"]
                                st.query_params["from_search"] = "false"
                                st.rerun()
# ---------- Search Results Page ----------
def show_search_results(q):
    st.subheader(f"🔍 10 Search Results for: `{q}`")
    if q:
        results = df[
            df['product_title'].str.contains(q, case=False, na=False) |
            df['product_description'].str.contains(q, case=False, na=False)
        ].drop_duplicates('parent_asin').head(10)
        if not results.empty:
            for row_items in [results.iloc[i:i+5] for i in range(0, len(results), 5)]:
                cols = st.columns(5)
                for idx, (_, item) in enumerate(row_items.iterrows()):
                    with cols[idx]:
                        with st.container(border=True):
                            img = load_and_pad_image(item['large_image_link'])
                            st.image(img)
                            st.caption(f"`{truncate_title(item['product_title'])}`")
                            if st.button("View", key=f"search_{item['parent_asin']}"):
                                st.query_params["page"] = "detail"
                                st.query_params["asin"] = item['parent_asin']
                                st.query_params["from_search"] = "true"
                                st.rerun()
        else:
            st.warning("No results found.")
    else:
        st.info("Please enter a search query.")
    if st.button("🔙 Back to Main Page"):
        st.session_state["reset_search"] = True
        st.query_params.clear()
        st.rerun()

# ---------- Route Pages ----------
if page == "detail" and selected_asin:
    show_product_detail(selected_asin)
elif page == "search":
    show_search_results(search_query)
else:
    show_home()
