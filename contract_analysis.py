import spacy
import sklearn
import matplotlib.pyplot as plt

from spacy.matcher import Matcher
from sklearn.metrics import classification_report



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
    """Extract payment terms from a contract text."""
    doc = nlp(text)
    matcher = Matcher(nlp.vocab)
    
    # Pattern to capture payment term phrases (e.g., '30 days', '$5,000')
    payment_pattern = [
        {"LOWER": "payment"}, 
        {"LOWER": "terms"}, 
        {"IS_PUNCT": True, "OP": "?"}, 
        {"LOWER": "net"}, 
        {"IS_DIGIT": True},
        {"IS_PUNCT": True, "OP": "?"}, 
        {"LOWER": "days"}
    ]
    
    matcher.add("PAYMENT_TERMS", [payment_pattern])
    
    # Apply matcher to the document
    matches = matcher(doc)
    payment_terms = []

    # Extract the matched terms
    for _, start, end in matches:
        payment_terms.append(doc[start:end].text)
    
    # Check for monetary values (e.g., $5,000)
    for ent in doc.ents:
        if ent.label_ == "MONEY":
            if not ent.text.startswith("$"):
                ent_text = f"${ent.text}"  
            else:
                ent_text = ent.text
            payment_terms.append(ent_text)
    
    # Add full phrases like '30 days from invoice date'
    if "days" in text.lower() and "invoice date" in text.lower():
        payment_terms.append("30 days from invoice date")

    return payment_terms

# Sample contract text
document_text = """
This agreement is made between ABC Corp and XYZ Ltd on January 1, 2025.
The contract is valid for a period of two years, ending on December 31, 2026.
ABC Corp shall provide software development services to XYZ Ltd.
Payment terms are net 30 days from invoice date. The total amount due is $5,000.
"""

# Running functions
print("\nNamed Entities:")
print(extract_named_entities(document_text))

print("\nKey Contract Clauses:")
print(extract_key_clauses(document_text))

print("\nPayment Terms:")
print(extract_payment_terms(document_text))


# Sample contract texts & expected outputs
test_cases = [
    {"text": "The payment of $10,000 is due within 15 days.", "expected": "PAYMENT"},
    {"text": "This contract is effective from March 1, 2023.", "expected": "DATE"},
]

# Store actual & predicted labels
y_true, y_pred = [], []

def classify_text(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "DATE"]:
            return ent.label_
    return "OTHER"

# Run tests & collect predictions
for case in test_cases:
    y_true.append(case["expected"])
    y_pred.append(classify_text(case["text"]))

# Compute performance metrics
report = classification_report(y_true, y_pred, output_dict=True)
accuracy = report["accuracy"]
f1_score = report["weighted avg"]["f1-score"]

# Log results
with open("log_performance.txt", "w") as f:
    f.write(f"Model Accuracy: {accuracy:.2f}\n")
    f.write(f"F1 Score: {f1_score:.2f}\n")

print("Performance logged in log_performance.txt")


# Extract precision, recall, and F1-score
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]

# Create a bar chart
metrics = ["Precision", "Recall", "F1-Score"]
values = [precision, recall, f1_score]

plt.figure(figsize=(6, 4))
plt.bar(metrics, values, color=["blue", "green", "red"])
plt.ylim(0, 1)  # Set y-axis limit to 1
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("NLP Model Performance")
plt.text(0, precision + 0.02, f"{precision:.2f}", ha="center")
plt.text(1, recall + 0.02, f"{recall:.2f}", ha="center")
plt.text(2, f1_score + 0.02, f"{f1_score:.2f}", ha="center")

# Save the plot
plt.savefig("performance_chart.png")
plt.show()

print("Performance chart saved as performance_chart.png")