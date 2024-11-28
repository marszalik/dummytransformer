import json
from HeadAttentionVectors import HeadAttentionVectors
class HeadAttention:
  EXISTENTIAL = "ExistentialWeigths.json"
  PRACTICAL = "PracticalWeights.json"
  DISTANCE_RELATED = "DistanceRelatedWeights.json"
  @staticmethod
  def multiply_matrices( a, b):
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert isinstance(a[0], list)
    assert isinstance(b[0], list)
    assert len(a[0]) == len(b)
    c = []
    for i in range(len(a)):
      c.append([])
      for j in range(len(b[0])):
        c[i].append(0)
        for k in range(len(b)):
          c[i][j] += a[i][k] * b[k][j]
    return c
    
  @staticmethod
  def trans_matrices( a):
    assert isinstance(a, list)
    assert isinstance(a[0], list)
    c = []
    for i in range(len(a[0])):
      c.append([])
      for j in range(len(a)):
        c[i].append(a[j][i])
    return c

  @staticmethod
  def add_matrices( a, b):
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert isinstance(a[0], list)
    assert isinstance(b[0], list)
    assert len(a) == len(b)
    assert len(a[0]) == len(b[0])
    c = []
    for i in range(len(a)):
      c.append([])
      for j in range(len(a[0])):
        c[i].append(a[i][j] + b[i][j])
    return c
  
  @staticmethod
  def scale_matrix( scalar, a):
    assert isinstance(a, list)
    assert isinstance(a[0], list)
    c = []
    for i in range(len(a)):
      c.append([])
      for j in range(len(a[0])):
        c[i].append(scalar * a[i][j])
    return c

  def __init__(self,head_name):
    with open(head_name) as json_file:
      head_weights = json.load(json_file)
      self.q_weights = head_weights["Q"]
      self.k_weights = head_weights["K"]
      self.v_weights = head_weights["V"]

  def calculate_Q(self, embeding):
    assert isinstance(embeding, list)
    assert len(embeding) == len(self.q_weights)
    return self.multiply_matrices([embeding], self.q_weights)

  def calculate_K(self, embeding):
    assert isinstance(embeding, list)
    assert len(embeding) == len(self.k_weights)
    return self.multiply_matrices([embeding], self.k_weights)

  def calculate_V(self, embeding):
    assert isinstance(embeding, list)
    assert len(embeding) == len(self.v_weights)
    return self.multiply_matrices([embeding], self.v_weights)

  @staticmethod
  def calculate_attention(headAttentionVectors: HeadAttentionVectors):
    assert isinstance(headAttentionVectors, HeadAttentionVectors)
    assert len(headAttentionVectors) > 0
    attentionModifiers = []
    for i in range(len(headAttentionVectors)):
      attentionModifiers.append([])
      q_vector = headAttentionVectors.get_q_matrix(i)
      for j in range(len(headAttentionVectors)):
        k_vector = headAttentionVectors.get_k_matrix(j)
        k_vector_trans = HeadAttention.trans_matrices(k_vector)
        a = HeadAttention.multiply_matrices(q_vector,k_vector_trans)
        attentionModifiers[i].append(a)
    attentionMatrix = []
    for i in range(len(attentionModifiers)):
      final_vector = [[0,0,0,0]]
      for j in range(len(attentionModifiers[0])):
        v_vector = headAttentionVectors.get_v_matrix(j)
        scaled_v_vector = HeadAttention.scale_matrix(attentionModifiers[i][j][0][0], v_vector)
        final_vector = HeadAttention.add_matrices(final_vector, scaled_v_vector)
        attentionMatrix.append(final_vector[0])
    return attentionMatrix
        
  pass