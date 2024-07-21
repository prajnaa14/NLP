from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity

# List of documents
documents = [
    "Shipment of gold damaged in a fire",
    "Delivery of silver arrived in a silver truck",
    "Shipment of gold arrived in a truck",
    "Purchased silver and gold arrived in a wooden truck",
    "The arrival of gold and silver shipment is delayed."
]

# Query document
query = "gold silver truck"

# Vectorize documents and query
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents).toarray()
query_vec = vectorizer.transform([query]).toarray().flatten()

# Calculate similarities
euclidean_distances = [euclidean(query_vec, doc) for doc in X]
manhattan_distances = [cityblock(query_vec, doc) for doc in X]
cosine_similarities = cosine_similarity([query_vec], X).flatten()

# Combine documents with their distances for ranking
documents_with_distances = list(zip(documents, euclidean_distances, manhattan_distances, cosine_similarities))

# Function to print top 2 relevant documents
def print_top_2(docs, key, reverse=False, measure_name=""):
    print(f"Top 2 relevant documents using {measure_name}:")
    sorted_docs = sorted(docs, key=key, reverse=reverse)[:2]
    for i, (doc, euclidean_dist, manhattan_dist, cosine_sim) in enumerate(sorted_docs, 1):
        value = key((doc, euclidean_dist, manhattan_dist, cosine_sim))
        print(f"{i}. Document: '{doc}'")
        print(f"   {measure_name}: {value:.4f}\n")

# Print results
print_top_2(documents_with_distances, key=lambda x: x[1], measure_name="Euclidean Distance")
print_top_2(documents_with_distances, key=lambda x: x[2], measure_name="Manhattan Distance")
print_top_2(documents_with_distances, key=lambda x: x[3], reverse=True, measure_name="Cosine Similarity")