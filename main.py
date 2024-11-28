from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding

from pprint import pprint


query = "what dog drinks"

words = query.split()

embeddings = Embeddings()
vectors =[]

for word in words:
    vector = embeddings.get_embedding(word)
    vectors.append(vector)

print("vektory embedingu")
pprint(vectors)


positionalEncoding = PositionalEncoding()
vectors_with_positions = positionalEncoding.add_position(vectors)

pprint(vectors_with_positions)

