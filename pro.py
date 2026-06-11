import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import matplotlib.pyplot as plt   # NEW IMPORT

# -------------------------
# PAGE SETTINGS
# -------------------------
st.set_page_config(page_title="Product Recommender", layout="wide")

# -------------------------
# SESSION STATE
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None
if "selected_product_index" not in st.session_state:
    st.session_state.selected_product_index = None
if "selected_preference" not in st.session_state:
    st.session_state.selected_preference = None
if "cart" not in st.session_state:
    st.session_state.cart = []

# -------------------------
# DATABASE SETUP
# -------------------------
conn = sqlite3.connect("ecommerce.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    product_title TEXT,
    product_category TEXT,
    price REAL,
    payment_method TEXT,
    order_date TEXT
)
""")

conn.commit()

# -------------------------
# LOAD DATASET
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("amazon_products_sales_data_cleaned.csv")
    df = df.fillna("")
    df["combined_features"] = df["product_title"].astype(str) + " " + df["product_category"].astype(str)
    df["discounted_price"] = pd.to_numeric(df["discounted_price"], errors="coerce").fillna(0)
    df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce").fillna(df["discounted_price"])
    return df

df = load_data()

# -------------------------
# TF-IDF MATRIX
# -------------------------
@st.cache_data
def compute_tfidf(data):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    return vectorizer.fit_transform(data["combined_features"])

tfidf_matrix = compute_tfidf(df)

# -------------------------
# PRODUCT RECOMMENDATIONS
# -------------------------
def recommend_products(selected_index, preference=None, num_recommendations=5):
    category_name = df.iloc[selected_index]["product_category"]

    category_df = df[df["product_category"].str.contains(category_name, case=False, na=False)]

    if preference:
        category_df = category_df[category_df["product_title"].str.contains(preference, case=False, na=False)]

    if category_df.empty:
        category_df = df[df["product_category"].str.contains(category_name, case=False, na=False)]

    original_indices = category_df.index.tolist()

    if selected_index not in original_indices and len(original_indices) > 0:
        selected_index = original_indices[0]

    product_vec = tfidf_matrix[selected_index]
    tfidf_subset = tfidf_matrix[original_indices]

    similarity_scores = cosine_similarity(product_vec, tfidf_subset).flatten()

    sorted_indices = similarity_scores.argsort()[::-1]

    top_indices = [original_indices[i] for i in sorted_indices if original_indices[i] != selected_index][:num_recommendations]

    return df.loc[top_indices]

# -------------------------
# COMPLETE SET
# -------------------------
def complete_set(selected_index, num_items=5):
    selected_product = df.iloc[selected_index]
    keywords = str(selected_product["product_title"]).split()

    other_products = df[df["product_category"] != selected_product["product_category"]]

    scores = []

    for idx, row in other_products.iterrows():
        match_count = sum([1 for k in keywords if k.lower() in str(row["product_title"]).lower()])
        scores.append((idx, match_count))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [idx for idx, _ in scores[:num_items]]

    return df.loc[top_indices]

# -------------------------
# LOGIN PAGE
# -------------------------
def login_page():

    st.markdown("""
    <style>

    .block-container{
        padding-top:3rem;
        max-width:500px;
        margin:auto;
    }

    div.stButton > button{
        background-color:#1f6feb;
        color:white;
        border-radius:8px;
        height:45px;
        font-size:18px;
        border:none;
    }

    div.stButton > button:hover{
        background-color:#1558c0;
        color:white;
    }

    </style>
    """, unsafe_allow_html=True)

    st.title("🔑 Login / Sign Up")

    username = st.text_input("Username", placeholder="Enter username", key="login_user")
    password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

    col1, col2 = st.columns([1,1])

    with col1:
        remember = st.checkbox("Remember me")

    with col2:
        st.markdown("<div style='text-align:right;color:#1f6feb;'>Forgot password?</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        login_btn = st.button("Login", use_container_width=True)

    with col2:
        signup_btn = st.button("Sign Up", use_container_width=True)

    if login_btn:

        if username and password:

            cursor.execute("SELECT password FROM users WHERE username=?", (username,))
            result = cursor.fetchone()

            if result:
                if result[0] == password:
                    st.session_state.user = username
                    st.session_state.page = "home"
                    st.success(f"Welcome back, {username}!")
                else:
                    st.error("Incorrect password!")
            else:
                st.warning("User does not exist. Please sign up first.")

        else:
            st.warning("Enter username and password")

    if signup_btn:

        if username and password:

            cursor.execute("SELECT * FROM users WHERE username=?", (username,))
            existing_user = cursor.fetchone()

            if existing_user:
                st.error("Username already exists. Try another one.")
            else:
                cursor.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, password))
                conn.commit()

                st.session_state.user = username
                st.session_state.page = "home"
                st.success(f"Account created successfully! Welcome {username}")

        else:
            st.warning("Enter username and password")

# -------------------------
# HOME PAGE
# -------------------------
def home_page():
    st.header(f"Hello, {st.session_state.user}!")
    st.write("Browse products and get smart recommendations.")

    st.sidebar.header("Product Filters")

    preference_options = ["All"]
    categories = df["product_category"].dropna().unique().tolist()
    categories.sort()

    selected_category = st.sidebar.selectbox("Select Category", categories, key="sidebar_category")

    if selected_category.lower() == "laptops":
        preference_options += ["Gaming", "Office"]
    elif selected_category.lower() == "smartphones":
        preference_options += ["Gaming", "Flagship"]
    elif selected_category.lower() == "tv":
        preference_options += ["Smart TV", "LED", "OLED"]

    selected_preference = st.sidebar.selectbox("Select Preference", preference_options, key="sidebar_preference")
    st.session_state.selected_preference = None if selected_preference == "All" else selected_preference

    category_df = df[df["product_category"].str.contains(selected_category, case=False, na=False)]

    if st.session_state.selected_preference:
        category_df = category_df[
            category_df["product_title"].str.contains(st.session_state.selected_preference, case=False, na=False)
        ]

    if category_df.empty:
        category_df = df[df["product_category"].str.contains(selected_category, case=False, na=False)]

    product_list = category_df["product_title"].tolist()
    selected_product_title = st.sidebar.selectbox("Choose Product", product_list, key="sidebar_product")

    selected_index = df[df["product_title"] == selected_product_title].index[0]
    st.session_state.selected_product_index = selected_index

    selected_row = df.iloc[selected_index]
    st.subheader(selected_row["product_title"])

    col1, col2 = st.columns([1, 2])

    with col1:
        if selected_row["product_image_url"]:
            st.image(selected_row["product_image_url"], width=250)
        else:
            st.text("No Image")

    with col2:
        st.write("Category:", selected_row["product_category"])
        st.write("⭐ Rating:", selected_row["product_rating"])
        st.write("💰 Price:", selected_row["discounted_price"])

        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            if st.button("Buy Now", key="buy_main"):
                st.session_state.cart.append(selected_row.to_dict())
                st.session_state.page = "checkout"

        with col_btn2:
            if st.button("Add to Cart", key="cart_main"):
                st.session_state.cart.append(selected_row.to_dict())
                st.success("Added to Cart!")

        # Styled cart button
        st.markdown("""
            <style>
            div.stButton > button[kind="secondary"] {
                font-weight: 700;
                background-color: #ff4b4b;
                color: white;
                border-radius: 10px;
                border: none;
                padding: 0.6rem 1rem;
            }
            div.stButton > button[kind="secondary"]:hover {
                background-color: #e63e3e;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)

        with col_btn3:
            if st.button("🛒 Go to Cart", key="goto_cart_main", type="secondary"):
                st.session_state.page = "cart"

    st.subheader("Similar Products")
    recommended = recommend_products(selected_index, st.session_state.selected_preference)

    if len(recommended) > 0:
        cols = st.columns(min(5, len(recommended)))
        for i, (_, row) in enumerate(recommended.iterrows()):
            with cols[i % min(5, len(recommended))]:
                if row["product_image_url"]:
                    st.image(row["product_image_url"], width=150)
                else:
                    st.write("No Image")
                st.write(str(row["product_title"])[:50] + "...")
                st.write("⭐", row["product_rating"])
                st.write("💰", row["discounted_price"])
                if st.button("Buy Now", key=f"buy_sim_{i}"):
                    st.session_state.cart.append(row.to_dict())
                    st.session_state.page = "checkout"
                if st.button("Add to Cart", key=f"cart_sim_{i}"):
                    st.session_state.cart.append(row.to_dict())
                    st.success("Added to Cart!")
    else:
        st.info("No similar products found.")

    st.subheader("Complete Your Set")
    complete_products = complete_set(selected_index)

    if len(complete_products) > 0:
        cols2 = st.columns(min(5, len(complete_products)))
        for i, (_, row) in enumerate(complete_products.iterrows()):
            with cols2[i % min(5, len(complete_products))]:
                if row["product_image_url"]:
                    st.image(row["product_image_url"], width=150)
                else:
                    st.write("No Image")
                st.write(str(row["product_title"])[:50] + "...")
                st.write("⭐", row["product_rating"])
                st.write("💰", row["discounted_price"])
                if st.button("Buy Now", key=f"buy_set_{i}"):
                    st.session_state.cart.append(row.to_dict())
                    st.session_state.page = "checkout"
                if st.button("Add to Cart", key=f"cart_set_{i}"):
                    st.session_state.cart.append(row.to_dict())
                    st.success("Added to Cart!")
    else:
        st.info("No complete set products found.")

# -------------------------
    # PRODUCT RECOMMENDATIONS
    # -------------------------
    st.subheader("Recommended Products")

    recommended_products = recommend_products(selected_index, st.session_state.selected_preference)

    for i, row in recommended_products.iterrows():

        col1, col2 = st.columns([1,3])

        with col1:
            if row["product_image_url"]:
                st.image(row["product_image_url"], width=120)

        with col2:
            st.write(row["product_title"])
            st.write("Price:", row["discounted_price"])

    # -------------------------
    # COMPLETE SET
    # -------------------------
    st.subheader("Frequently Bought Together")

    complete_products = complete_set(selected_index)

    for i, row in complete_products.iterrows():

        col1, col2 = st.columns([1,3])

        with col1:
            if row["product_image_url"]:
                st.image(row["product_image_url"], width=120)

        with col2:
            st.write(row["product_title"])
            st.write("Price:", row["discounted_price"])

# -------------------------
# CART PAGE
# -------------------------
def cart_page():
    st.header("🛒 Your Cart")
    if not st.session_state.cart:
        st.info("Your cart is empty!")
        if st.button("Go Back to Home"):
            st.session_state.page = "home"
        return

    for i, product in enumerate(st.session_state.cart):
        st.subheader(product["product_title"])
        col1, col2 = st.columns([1, 2])
        with col1:
            if product["product_image_url"]:
                st.image(product["product_image_url"], width=150)
        with col2:
            original_price = float(product.get("original_price", product["discounted_price"]))
            discounted_price = float(product["discounted_price"])
            st.write("Original Price: ₹", original_price)
            st.write("Discounted Price: ₹", discounted_price)

            enable_alert = st.checkbox(f"Enable Price Drop Alert for {product['product_title']}", key=f"alert_{i}")
            if enable_alert:
                if discounted_price < original_price:
                    st.success(f"Price Dropped! Now: ₹{discounted_price} (was ₹{original_price})")
                else:
                    st.error(f"No Price Drop. Current: ₹{discounted_price} (Original: ₹{original_price})")

            if st.button(f"Buy Now - {product['product_title']}", key=f"buy_cart_{i}"):
                st.session_state.selected_product_index = df[df["product_title"] == product["product_title"]].index[0]
                st.session_state.page = "checkout"

    if st.button("Go Back to Home"):
        st.session_state.page = "home"

# -------------------------
# CHECKOUT PAGE
# -------------------------
def checkout_page():
    st.header("🛒 Checkout")
    product = st.session_state.cart[0] if st.session_state.cart else df.iloc[st.session_state.selected_product_index]
    st.subheader(product["product_title"])

    # UPDATED FIELD 2: show dollar and rupees
    price_dollar = float(product["discounted_price"])
    rupee_value = price_dollar * 83
    st.write(f"Price: ${price_dollar:.2f}")
    st.write(f"Price in Rupees: ₹{rupee_value:,.2f}")

    st.write("Enter shipping details:")
    name = st.text_input("Full Name", key="ship_name")
    address = st.text_area("Address", key="ship_addr")
    city = st.text_input("City", key="ship_city")
    zip_code = st.text_input("ZIP / Postal Code", key="ship_zip")

    st.write("Select Payment Method:")
    payment = st.selectbox("Payment Method", ["Credit/Debit Card", "UPI", "Net Banking", "Cash on Delivery"], key="pay_method")

    if payment == "Credit/Debit Card":
        card_number = st.text_input("Card Number", key="card_num")
        expiry = st.text_input("Expiry Date (MM/YY)", key="card_exp")
    elif payment == "UPI":
        upi_id = st.text_input("Enter UPI ID", key="upi_id")
    elif payment == "Net Banking":
        bank = st.selectbox("Select Bank", ["HDFC", "SBI", "ICICI", "Axis"], key="bank_sel")

    if st.button("Place Order"):
        if not all([name, address, city, zip_code]):
            st.warning("Please fill all shipping details!")
            return

        cursor.execute("""
            INSERT INTO orders (username, product_title, product_category, price, payment_method, order_date)
            VALUES (?,?,?,?,?,?)
        """, (
            st.session_state.user,
            product["product_title"],
            product["product_category"],
            product["discounted_price"],
            payment,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        conn.commit()
        st.session_state.cart = []
        st.session_state.page = "thankyou"
# -------------------------
# THANK YOU
# -------------------------
def thankyou_page():
    st.balloons()
    st.success("🎉 Thank you for your order! Your order has been placed successfully.")

    # Wait for 3 seconds
    time.sleep(3)

    # Redirect to home page automatically
    st.session_state.page = "home"
    st.rerun()
# -------------------------
# PAGE ROUTER
# -------------------------
if st.session_state.page=="login":
    login_page()

elif st.session_state.page=="home":
    home_page()

elif st.session_state.page=="cart":
    cart_page()

elif st.session_state.page=="checkout":
    checkout_page()

elif st.session_state.page=="thankyou":
    thankyou_page()