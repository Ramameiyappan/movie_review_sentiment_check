# Movie Review Sentiment Analysis

This project is a **Movie Review Sentiment Analysis** application built using **Natural Language Processing (NLP)** techniques.  
It uses **TF-IDF** for feature extraction and **Logistic Regression** for sentiment classification, deployed using **Streamlit**.

🔗 **Live App**  
https://moviereviewsentimentcheck.streamlit.app/

---

## 📂 Project Structure
```
project/
│
├── icon/                         # Folder containing emojis/icons used in UI
│
├── app.py                        # Main Streamlit application
│
├── logistic_tfidf.pkl            # Trained Logistic Regression model
├── tfidf_vectorizer.pkl          # Trained TF-IDF Vectorizer
│
├── requirements.txt              # Project dependencies
```

---

## 🚀 Features

- Analyze movie reviews and predict **Positive** or **Negative** sentiment
- Simple and interactive **Streamlit UI**
- NLP preprocessing using **spaCy**
- Lightweight ML model (**No transformers used**)
- Fast inference using pre-trained `.pkl` models

---

## 🧠 Model & NLP Pipeline

### 1️⃣ Text Preprocessing
- Lowercasing
- Tokenization
- Stopword removal
- Lemmatization using spaCy (`en_core_web_sm`)

### 2️⃣ Feature Extraction
- TF-IDF Vectorization

### 3️⃣ Model
- Logistic Regression Classifier

### 4️⃣ Output
- `1` → Positive  
- `0` → Negative  

---

## 📦 Requirements

```bash
streamlit
numpy
joblib==1.5.3
scikit-learn==1.7.2
spacy==3.8.11
https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz#egg=en_core_web_sm
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ramameiyappan/movie_review_sentiment_check.git
cd movie_review_sentiment_check
```

---

### 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv nlp
```

---

### 3️⃣ Activate Virtual Environment

**Linux / macOS**

```bash
source nlp/bin/activate
```

**Windows**

```bash
nlp\Scripts\activate
```

---

### 4️⃣ Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---
## 🛠 Technologies Used

```bash
Python
Streamlit
Scikit-learn
spaCy
TF-IDF
Logistic Regression
Joblib
```
