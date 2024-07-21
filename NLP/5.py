from collections import defaultdict
import numpy as np

# Data
documents = [("fun, couple, love, love", "comedy"),
             ("fast, furious, shoot", "action"),
             ("couple, fly, fast, fun, fun", "comedy"),
             ("furious, shoot, shoot, fun", "action"),
             ("fly, fast, shoot, love", "action")]
test = "fast, couple, shoot, fly"

# Count words and classes
word_counts = defaultdict(lambda: defaultdict(int))
class_counts = defaultdict(int)
vocab = set()

for doc, label in documents:
    words = doc.lower().split(",")
    for word in words:
        word_counts[label][word] += 1
        vocab.add(word)
    class_counts[label] += 1

# Calculate priors and likelihoods
total_docs = sum(class_counts.values())
vocab_size = len(vocab)
priors = {label: count / total_docs for label, count in class_counts.items()}
likelihood = lambda word, label: (word_counts[label][word] + 1) / (sum(word_counts[label].values()) + vocab_size)

# Compute posterior
def compute_posterior(doc, label):
    words = doc.lower().split(", ")
    return np.log(priors[label]) + sum(np.log(likelihood(word, label)) for word in words)

# Classify new document
predicted_class = max(priors, key=lambda label: compute_posterior(test, label))
print(f"The most likely class for the new document '{test}' is: {predicted_class}")
