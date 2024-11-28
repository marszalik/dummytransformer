from HeadAttention import HeadAttention
from HeadAttentionVectors import HeadAttentionVectors
from pprint import pprint

print("test")
e = [[1,2,3,4]]
e1 = [[5,6,7,8]]
x = HeadAttention.add_matrices(e, e1)
pprint(x)
x = HeadAttention.scale_matrix(2, e)
pprint(x)
w = [[5,6],[7,8],[9,10],[11,12]]
x = HeadAttention.multiply_matrices(e, w)
print("test")
pprint(x)
x = HeadAttention.trans_matrices(e)
pprint(x)
existential_head = HeadAttention(HeadAttention.EXISTENTIAL)
practical_head = HeadAttention(HeadAttention.PRACTICAL)
distance_related_head = HeadAttention(HeadAttention.DISTANCE_RELATED)
e = [1,2,3,4]
q = existential_head.calculate_Q(e)
pprint(q)
k = existential_head.calculate_K(e)
pprint(k)
v = existential_head.calculate_V(e)
pprint(v)
hav = HeadAttentionVectors()
hav.append(q, k, v)
e = [5,6,7,8]
q = existential_head.calculate_Q(e)
pprint(q)
k = existential_head.calculate_K(e)
pprint(k)
v = existential_head.calculate_V(e)
pprint(v)
hav.append(q, k, v)
attentionMatrix = HeadAttention.calculate_attention(hav)
pprint(attentionMatrix)