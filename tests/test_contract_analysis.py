import pytest
from contract_analysis import extract_named_entities, extract_payment_terms

# Sample contract text for testing
sample_text = """
This Agreement is made on January 1, 2024, between Zenlaw Corp. and ABC Legal Services.
The payment of $5,000 is due within 30 days from the invoice date.
"""

def test_named_entity_extraction():
    entities = extract_named_entities(sample_text)
    
    assert "Zenlaw Corp." in entities.get("ORG", [])
    assert "January 1, 2024" in entities.get("DATE", [])

def test_payment_term_extraction():
    payment_terms = extract_payment_terms(sample_text)
    
    assert "30 days from invoice date" in payment_terms
    assert "$5,000" in payment_terms  

if __name__ == "__main__":
    pytest.main()
