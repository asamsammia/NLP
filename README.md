# NLP Contract Analysis with SpaCy & Transformers

## Overview
This project demonstrates how to use **Natural Language Processing (NLP)** to analyze legal contracts. Using **SpaCy, Transformers, and Python**, we extract key clauses and automate contract review.

## Features
✅ Named Entity Recognition (NER) for legal terms  
✅ Clause classification using Machine Learning  
✅ Automated text preprocessing and tokenization  
✅ Visualization of contract structures  

## Tech Stack
- **Programming Language:** Python
- **NLP Frameworks:** SpaCy, Transformers (Hugging Face)
- **Libraries:** Pandas, Scikit-learn, Matplotlib

## Installation
```bash
# Clone the repository
git clone https://github.com/your-username/nlp-contract-analysis.git
cd nlp-contract-analysis

# Create a virtual environment
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Dataset
- Uses publicly available legal contract datasets
- Preprocessed for entity recognition and classification

## Usage
```python
from spacy.lang.en import English
import spacy

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Process a sample contract
text = "This agreement is made between Party A and Party B effective January 1, 2025."
doc = nlp(text)

# Extract Named Entities
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## Expected Output
```
Party A ORG
Party B ORG
January 1, 2025 DATE
```

## Next Steps
🔹 Expand entity recognition with custom NER models  
🔹 Train a transformer-based model for clause classification  
🔹 Deploy as a REST API for automated document analysis  

## Contributing
Feel free to **fork** this repo, open **issues**, and submit **pull requests**!

## License
MIT License
