# NLP Contract Analysis

## 📌 Overview
This repository contains a **Natural Language Processing (NLP) project** focused on analyzing legal contracts. Using **SpaCy, Transformers, and Python**, we automate the extraction of key clauses, named entities, and payment terms to streamline contract review processes.

## 🚀 Features
- **Named Entity Recognition (NER):** Identifies companies, dates, and contract-specific terms.
- **Clause Classification:** Extracts important legal clauses using NLP models.
- **Rule-Based Matching:** Detects payment terms and obligations.
- **Data Visualization:** Displays named entity distributions using Matplotlib.

## 🛠️ Tech Stack
- **Programming Language:** Python
- **NLP Libraries:** SpaCy, Transformers
- **Data Processing:** Pandas, Scikit-learn
- **Visualization:** Matplotlib

## 📂 Project Structure
```
nlp-contract-analysis/
│── data/                # Sample contract datasets
│── notebooks/           # Jupyter Notebook for analysis
│── scripts/             # Python scripts for NLP processing
│── README.md            # Project Documentation
│── requirements.txt     # Dependencies
```

## 🔧 Installation
1️⃣ Clone the repository:
```bash
git clone https://github.com/your-username/nlp-contract-analysis.git
cd nlp-contract-analysis
```
2️⃣ Create a virtual environment and install dependencies:
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
```

## 🏃 Usage
```python
from spacy.lang.en import English
import spacy

nlp = spacy.load("en_core_web_sm")
text = "This agreement is made between Party A and Party B effective January 1, 2025."
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```
**Expected Output:**
```
Party A (ORG)
Party B (ORG)
January 1, 2025 (DATE)
```

## 📊 Visualization
To analyze entity distribution:
```python
plt.figure(figsize=(8, 4))
df_entities["Label"].value_counts().plot(kind="bar", color="skyblue")
plt.title("Named Entity Distribution")
plt.xlabel("Entity Type")
plt.ylabel("Count")
plt.show()
```

## 🎯 Next Steps
- Fine-tune a **Transformer-based NER model** for better accuracy.
- Deploy the model as an API for **automated contract analysis**.

## 📜 License
MIT License

---
🚀 *Feel free to fork, contribute, and open issues!*
