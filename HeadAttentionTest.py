from HeadAttention import HeadAttention
from pprint import pprint

print("test")
e = [[1,2,3,4]]
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