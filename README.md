# 🌍 Tourism Review Sentiment Analysis (End-to-End Pipeline)

![](/resources/traveloka-head.jpg)

![](/resources/agoda-head.jpg)

## 📌 Overview

This project builds an **end-to-end data science & NLP pipeline** for **sentiment analysis on tourism service reviews** collected from popular online travel platforms.

The goal is to **analyze customer opinions and sentiments** from real-world reviews, combining:

* Web crawling
* Statistical hypothesis testing
* Multilingual text embedding
* Machine Learning & Neural Networks

Raw data is stored in a **NoSQL MongoDB database**, reflecting a production-oriented data architecture.

---

## 🏗️ Project Pipeline

1. **Web Crawling & Data Collection**
2. **Raw Data Storage (MongoDB)**
3. **Data Cleaning & Preprocessing**
4. **Exploratory Data Analysis (EDA)**
5. **Statistical Hypothesis Testing**
6. **Text Embedding with LaBSE**
7. **Sentiment Classification (ML & Neural Networks)**
8. **Model Evaluation**

---

## 🌐 Data Collection (Web Crawling)

### Data Sources

* **Agoda** – tourism service listings and customer reviews
* **Traveloka** – accommodation and travel service reviews

### Crawling Tools

* **BeautifulSoup**: HTML parsing and static content extraction
* **Selenium**: handling dynamic content, pagination, JavaScript-rendered reviews, and user interactions

### Collected Data

* Service name
* Location
* Review text
* Rating score
* Review date
* Platform source (Agoda / Traveloka)

---

## 🗄️ Data Storage

### NoSQL Database

* **MongoDB** is used to store **raw crawled data**

### Benefits

* Flexible schema for heterogeneous review data
* Easy scalability for large volumes of text data
* Clear separation between raw and processed datasets

---

## 🔍 Exploratory Data Analysis (EDA)

EDA is performed to understand review characteristics and sentiment trends.

### EDA Tasks

* Review length distribution
* Rating distribution across platforms
* Sentiment polarity vs rating score
* Platform-level comparison (Agoda vs Traveloka)

### Tools

* Pandas
* NumPy
* Matplotlib
* Seaborn

---

## 📐 Statistical Hypothesis Testing

To validate assumptions and insights, statistical tests are applied.

### Libraries

* **SciPy**
* **Statsmodels**

### Example Hypotheses

* Difference in average ratings between platforms
* Relationship between rating score and sentiment polarity
* Statistical significance of sentiment differences across locations

Common techniques:

* t-test
* ANOVA
* Correlation analysis
* Confidence intervals

---

## 🧹 Data Preprocessing

### Text Processing

* Text normalization (lowercasing, punctuation removal)
* Noise removal (HTML tags, emojis if needed)
* Handling multilingual reviews

### Label Preparation

* Sentiment labels derived from ratings (Positive / Neutral / Negative)
* Class balancing (if applicable)

---

## 🧠 Text Embedding

### Embedding Model

* **LaBSE (Language-agnostic BERT Sentence Embedding)**

### Purpose

* Generate **language-independent sentence embeddings**
* Enable multilingual sentiment analysis across different review languages

### Tools

* Hugging Face **transformers**
* PyTorch (**torch**)

---

## 🤖 Sentiment Analysis Models

Both traditional machine learning and neural network approaches are explored.

### Machine Learning Models

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

### Neural Network Models

* Feed-forward Neural Networks
* Transformer-based fine-tuning (if applicable)

### Libraries

* **Scikit-learn**
* **PyTorch (torch)**
* **Transformers**

---

## 📊 Model Evaluation

### Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix

Models are compared to assess:

* Effectiveness of LaBSE embeddings
* Performance difference between ML and NN approaches

---

## 🛠️ Tech Stack

| Category        | Tools                   |
| --------------- | ----------------------- |
| Crawling        | BeautifulSoup, Selenium |
| Database        | MongoDB (NoSQL)         |
| Data Processing | Pandas, NumPy           |
| Visualization   | Matplotlib, Seaborn     |
| Statistics      | SciPy, Statsmodels      |
| NLP             | LaBSE, Transformers     |
| ML & DL         | Scikit-learn, PyTorch   |
| Language        | Python                  |

---

## 📁 Project Structure

```
tourism-sentiment-analysis/
│
├── data/
│   ├── raw/              # Raw data exported from MongoDB
│   ├── processed/        # Cleaned and labeled data
│
├── crawling/
│   ├── agoda_crawler.py
│   └── traveloka_crawler.py
│
├── database/
│   └── mongodb_utils.py
│
├── notebooks/
│   ├── eda.ipynb
│   ├── hypothesis_testing.ipynb
│   └── modeling.ipynb
│
├── models/
│   └── sentiment_models.pt
│
├── README.md
└── requirements.txt
```

---

## 🚀 Future Improvements

* Aspect-based sentiment analysis (price, service, location)
* Topic modeling for customer feedback insights
* Real-time review ingestion
* Deployment as an API or dashboard

---

## 👤 Author

**Nguyễn Huỳnh Minh Phú**
Data Science / NLP Student

---

## 📄 License

This project is for educational and research purposes.
