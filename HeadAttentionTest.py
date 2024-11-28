from HeadAttention import HeadAttention
from pprint import pprint

print("test")
v = [[1,2,3,4]]
w = [[5,6],[7,8],[9,10],[11,12]]
x = HeadAttention.multiply_matrices(v, w)
print("test")
pprint(x)
x = HeadAttention.trans_matrices(v)
pprint(x)