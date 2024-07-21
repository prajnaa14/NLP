import nltk
from nltk import CFG
from nltk.parse import RecursiveDescentParser, ShiftReduceParser
 
# Define a simple context-free grammar
grammar = CFG.fromstring("""
    S -> NP VP
    NP -> DT NN
    VP -> V NP
    DT -> 'the'
    NN -> 'cat' | 'dog' | 'telescope' | 'park'
    V -> 'saw' | 'chased'
""")

# Example sentence
sentence = "the cat saw the dog".split()

# Function to parse and print the parse trees
def parse_and_print(parser, sentence, parser_name):
    print(f"\n{parser_name} Parsing:")
    for tree in parser.parse(sentence):
        print(tree)
        tree.pretty_print()

# Top-Down Parsing using RecursiveDescentParser
parse_and_print(RecursiveDescentParser(grammar), sentence, "Top-Down")

# Bottom-Up Parsing using ShiftReduceParser
parse_and_print(ShiftReduceParser(grammar), sentence, "Bottom-Up")