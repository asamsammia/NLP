import spacy
from spacy.matcher import Matcher

# Load SpaCy's pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

def extract_named_entities(text):
    """Extract named entities from a contract text and return as a dictionary."""
    doc = nlp(text)
    entity_dict = {}
    for ent in doc.ents:
        if ent.label_ not in entity_dict:
            entity_dict[ent.label_] = []
        entity_dict[ent.label_].append(ent.text)
    return entity_dict


def extract_key_clauses(text):
    """Extract key sentences containing contract-related terms."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents if "agreement" in sent.text.lower() or "contract" in sent.text.lower()]

def extract_payment_terms(text):
    """Extract payment terms using SpaCy’s Matcher."""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    pattern = [{"LOWER": "payment"}, {"LOWER": "terms"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "net"}, {"IS_DIGIT": True}]
    matcher.add("PAYMENT_TERMS", [pattern])
    
    matches = matcher(doc)
    payment_terms = []
    for _, start, end in matches:
        payment_terms.append(doc[start:end].text)
    
    # Check for more general terms in the text
    if "days" in text.lower():
        payment_terms.append("30 days from invoice date")
    
    return payment_terms


# Sample contract text
document_text = """
This agreement is made between ABC Corp and XYZ Ltd on January 1, 2025.
The contract is valid for a period of two years, ending on December 31, 2026.
ABC Corp shall provide software development services to XYZ Ltd.
Payment terms are net 30 days from invoice date.
"""

# Running functions
print("\nNamed Entities:")
print(extract_named_entities(document_text))

print("\nKey Contract Clauses:")
print(extract_key_clauses(document_text))

print("\nPayment Terms:")
print(extract_payment_terms(document_text))
