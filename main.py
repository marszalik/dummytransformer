from Embeddings import Embeddings
from PositionalEncoding import PositionalEncoding
from HeadAttention import HeadAttention
from HeadAttentionVectors import HeadAttentionVectors
from Multihead import Multihead

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

#Existential head    
headAttention = HeadAttention(HeadAttention.EXISTENTIAL)
hav = HeadAttentionVectors()
for i in range(len(vectors_with_positions)):
    q_vector = headAttention.calculate_Q(vectors_with_positions[i])
    k_vector = headAttention.calculate_K(vectors_with_positions[i])
    v_vector = headAttention.calculate_V(vectors_with_positions[i])        
    hav.append(q_vector, k_vector, v_vector)
existentialAttentionMatrix = headAttention.calculate_attention(hav)
pprint(existentialAttentionMatrix)

#Practical head    
headAttention = HeadAttention(HeadAttention.PRACTICAL)
hav = HeadAttentionVectors()
for i in range(len(vectors_with_positions)):
    q_vector = headAttention.calculate_Q(vectors_with_positions[i])
    k_vector = headAttention.calculate_K(vectors_with_positions[i])
    v_vector = headAttention.calculate_V(vectors_with_positions[i])        
    hav.append(q_vector, k_vector, v_vector)
practicalAttentionMatrix = headAttention.calculate_attention(hav)
pprint(practicalAttentionMatrix)

#Distance related head    
headAttention = HeadAttention(HeadAttention.DISTANCE_RELATED)
hav = HeadAttentionVectors()
for i in range(len(vectors_with_positions)):
    q_vector = headAttention.calculate_Q(vectors_with_positions[i])
    k_vector = headAttention.calculate_K(vectors_with_positions[i])
    v_vector = headAttention.calculate_V(vectors_with_positions[i])        
    hav.append(q_vector, k_vector, v_vector)
distanceRelatedAttentionMatrix = headAttention.calculate_attention(hav)
pprint(distanceRelatedAttentionMatrix)

multihead = Multihead()
multihead.add_outupt_from_head(existentialAttentionMatrix)
multihead.add_outupt_from_head(practicalAttentionMatrix)
multihead.add_outupt_from_head(distanceRelatedAttentionMatrix)
multiheadOutput = multihead.combineHeads()
pprint(multiheadOutput)