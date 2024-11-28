from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from HeadAttention import HeadAttention
from HeadAttentionVectors import HeadAttentionVectors

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

for head in [HeadAttention.EXISTENTIAL, HeadAttention.PRACTICAL, HeadAttention.DISTANCE_RELATED]:
    headAttention = HeadAttention(head)
    hav = HeadAttentionVectors()
    for i in range(len(vectors_with_positions)):
        q_vector = headAttention.calculate_Q(vectors_with_positions[i])
        k_vector = headAttention.calculate_K(vectors_with_positions[i])
        v_vector = headAttention.calculate_V(vectors_with_positions[i])        
        hav.append(q_vector, k_vector, v_vector)
    attentionMatrix = headAttention.calculate_attention(hav)
    pprint(attentionMatrix)
    print("test")