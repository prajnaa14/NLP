import nltk
from nltk.corpus import wordnet

# Download the WordNet data
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_synonyms_antonyms(word):
    synonyms = []
    antonyms = []
    
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    
    # Remove duplicates
    synonyms = set(synonyms)
    antonyms = set(antonyms)
    
    return synonyms, antonyms

# Example usage
word = input("Enter the word: ")
synonyms, antonyms = get_synonyms_antonyms(word)

print(f"Synonyms of '{word}': {synonyms}")
print(f"Antonyms of '{word}': {antonyms}")