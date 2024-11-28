from HeadAttention import HeadAttention
from HeadAttentionVectors import HeadAttentionVectors
from pprint import pprint

# private methods tests
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


# Head attention tests
# Head creation
existential_head = HeadAttention(HeadAttention.EXISTENTIAL)
practical_head = HeadAttention(HeadAttention.PRACTICAL)
distance_related_head = HeadAttention(HeadAttention.DISTANCE_RELATED)
# test embeding
e = [1,2,3,4]
# q,k,v calculation
q = existential_head.calculate_Q(e)
pprint(q)
k = existential_head.calculate_K(e)
pprint(k)
v = existential_head.calculate_V(e)
pprint(v)
# q,k,v vector collection
hav = HeadAttentionVectors()
hav.append(q, k, v)
# q,k,v calculation for different embeding
e = [5,6,7,8]
q = existential_head.calculate_Q(e)
pprint(q)
k = existential_head.calculate_K(e)
pprint(k)
v = existential_head.calculate_V(e)
pprint(v)
hav.append(q, k, v)
# attention calculation
attentionMatrix = HeadAttention.calculate_attention(hav)
pprint(attentionMatrix)