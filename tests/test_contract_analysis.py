python3 -m pip install pytest spacy

import spacy
import pytest
from contract_analysis import extract_named_entities, extract_payment_terms

# Load SpaCy model (ensure you have the right model installed)
nlp = spacy.load("en_core_web_sm")

# Sample contract text for testing
sample_text = """
This Agreement is made on January 1, 2024, between Zenlaw Corp. and ABC Legal Services.
The payment of $5,000 is due within 30 days from the invoice date.
"""

def test_named_entity_extraction():
    entities = extract_named_entities(sample_text, nlp)
    assert "Zenlaw Corp." in entities["ORG"]
    assert "January 1, 2024" in entities["DATE"]

def test_payment_term_extraction():
    payment_terms = extract_payment_terms(sample_text, nlp)
    assert "30 days from the invoice date" in payment_terms
    assert "$5,000" in payment_terms

if __name__ == "__main__":
    pytest.main()
