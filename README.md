# 📧 Spam Email Classification (NLP)

A machine learning project to detect spam emails using text preprocessing, feature extraction, and classification algorithms. Built with **Python, Scikit-Learn, and NLP techniques**.

---

## 🔍 Project Overview
This project applies **Natural Language Processing (NLP)** to classify emails as **Spam** or **Ham (Not Spam)**.  
We experimented with **Naïve Bayes, Support Vector Machines (SVM), and Logistic Regression**.  
By applying **text normalization and TF-IDF vectorization**, the **SVM model achieved 91% accuracy** with an **AUC score of 0.97**, outperforming other approaches.

---

## 🛠️ Tech Stack
- **Languages**: Python  
- **Libraries**: NumPy, Pandas, Scikit-Learn, NLTK  
- **Techniques**: Text Cleaning, Tokenization, TF-IDF Vectorization  
- **Models**:  
  - Logistic Regression  
  - Naïve Bayes  
  - Support Vector Machine (SVM)  

---

## 📂 Repository Structure
```
├── Spam_Classification_NLP.ipynb   # Jupyter notebook with end-to-end workflow
├── NLP_Report.pdf                  # Project report / documentation
└── README.md                       # Project overview (this file)
```

---

## 🚀 Workflow
1. **Data Preprocessing**
   - Text normalization (lowercasing, punctuation removal, stopword filtering).  
   - Tokenization and lemmatization for cleaner text features.  

2. **Feature Engineering**
   - Applied **TF-IDF vectorization** to convert text into numerical features.  

3. **Modeling**
   - Trained and evaluated Logistic Regression, Naïve Bayes, and SVM.  
   - Compared metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.  

4. **Results**
   - **SVM** performed best with **91% accuracy** and **0.97 ROC-AUC**.

---

## 📊 Results Snapshot
| Model                | Accuracy | ROC-AUC |
|----------------------|----------|---------|
| Logistic Regression  | 88%      | 0.94    |
| Naïve Bayes          | 89%      | 0.95    |
| **SVM**              | **91%**  | **0.97** |

---

## 📖 Key Learnings
- Importance of preprocessing text data for NLP.  
- How TF-IDF balances word importance in large corpora.  
- SVM’s effectiveness on high-dimensional feature spaces.  

---

## 📌 Future Improvements
- Try **deep learning models** (LSTMs, Transformers).  
- Use **word embeddings (Word2Vec, GloVe, BERT)** for richer features.  
- Deploy as a **Flask/Streamlit web app** for live spam detection.  

---

## 👤 Author
**Akanksh Shetty**  
- [LinkedIn](https://www.linkedin.com/in/akanksh17)  
- [Portfolio](https://akanksh171717.github.io)  
- [GitHub](https://github.com/Akanksh171717)  
