import streamlit as st
import joblib
import spacy
import random 

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("logistic_tfidf.pkl")
    return nlp, tfidf, model

nlp, tfidf, logistic_tfidf = load_models()

def predict_sentiment(review):
    doc = nlp(review)
    tokens = [token.lemma_ for token in doc if not token.is_punct]
    tfidf_review = [' '.join(tokens)]
    vector = tfidf.transform(tfidf_review)
    prediction = logistic_tfidf.predict(vector)[0]
    return prediction


st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="icon/logo.png",
)

st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
}

h1 {
    color: #ff416c;
    font-weight: 800;
}

h3 {
    color: #333333;
}

textarea {
    border-radius: 14px;
    border: 2px solid #764ba2;
    font-size: 16px;
    padding: 12px;
}

textarea:focus {
    border-color: #ff416c;
    box-shodow: 0 0 10px rgba(255,65,108,0.4);
}

.stButton button {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
    color: white;
    border-radius: 14px;
    height: 55px;
    font-size: 19px;
    font-weight: bold;
    letter-spacing: 0.5px;
    transition: all 0.3s ease-in-out;
    box-shadow: 0px 8px 20px rgba(255,75,75,0.4);
}

.stButton button:hover {
    background: linear-gradient(90deg, #ff4b2b, #ff416c);
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0px 12px 30px rgba(255,75,75,0.6);
}

.result-positive {
    display: inline-block;
    background: linear-gradient(90deg, #56ab2f, #a8e063);
    color: white;
    padding: 12px 20px;
    border-radius: 14px;
    font-weight: bold;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

.result-negative {
    display: inline-block;
    background: linear-gradient(90deg, #cb2d3e, #ef473a);
    color: white;
    padding: 12px 20px;
    border-radius: 14px;
    font-weight: bold;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
}

.footer {
    text-align: center;
    font-size: 13px;
    color: #eeeeee;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align: center;'>Movie Review Sentiment Analysis</h1>
<p style='text-align: center; color: gray; font-size:16px;'>
Analyze whether a movie review expresses a <b>Positive</b> or <b>Negative</b> sentiment
</p>
<hr>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 10])
no = random.randint(1,5)
with col1:
    st.image(f"icon/writing{no}.png")
with col2:
    st.markdown("### Enter Movie Review")

review_text = st.text_area(
    "",
    height=180,
    placeholder="Example: The movie was visually stunning and emotionally powerful..."
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3= st.columns([1, 2, 1])
with col2:
    predict_btn = st.button("Predict Sentiment", use_container_width=True)

if predict_btn:
    if review_text.strip() == "":
        st.warning("Please enter a movie review before predicting.")
    else:
        result = predict_sentiment(review_text.lower().strip())

        st.markdown("<br>", unsafe_allow_html=True)

        if result == 1:
            col_img, col_text = st.columns([1, 5])
            with col_img:
                st.image("icon/positive.png", width=60)
            with col_text:
                st.markdown("<div class='result-positive'>Positive Review</div>", unsafe_allow_html=True)
            st.markdown("This review expresses **favorable opinions** about the movie")
        else:
            col_img, col_text = st.columns([1, 5])
            with col_img:
                st.image("icon/negative.png", width=60)
            with col_text:
                st.markdown("<div class='result-negative'>Negative Review</div>", unsafe_allow_html=True)
            st.markdown("This review expresses **unfavorable opinions** about the movie.")

st.markdown("""
<hr>
<p style='text-align:center; font-size:13px; color:gray;'>
Built using <b>TF-IDF + Logistic Regression</b> | NLP Project
</p>
""", unsafe_allow_html=True)