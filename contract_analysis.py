import spacy

# Load SpaCy's pre-trained NLP model
nlp = spacy.load("en_core_web_sm")

# Sample contract text
document_text = """
This agreement is made between ABC Corp and XYZ Ltd on January 1, 2025.
The contract is valid for a period of two years, ending on December 31, 2026.
ABC Corp shall provide software development services to XYZ Ltd.
Payment terms are net 30 days from invoice date.
"""

# Process the text
doc = nlp(document_text)

# Extract named entities
print("\nNamed Entities:")
for ent in doc.ents:
    print(f"{ent.text} ({ent.label_})")

# Extract key sentences for contract analysis
important_phrases = [sent.text for sent in doc.sents if "agreement" in sent.text.lower() or "contract" in sent.text.lower()]

print("\nKey Contract Clauses:")
for clause in important_phrases:
    print(clause)

# Example of rule-based matching (customization can be added)
from spacy.matcher import Matcher

matcher = Matcher(nlp.vocab)
pattern = [{"LOWER": "payment"}, {"LOWER": "terms"}, {"IS_PUNCT": True, "OP": "?"}, {"LOWER": "net"}, {"IS_DIGIT": True}]
matcher.add("PAYMENT_TERMS", [pattern])

matches = matcher(doc)
print("\nPayment Terms:")
for match_id, start, end in matches:
    print(doc[start:end].text)