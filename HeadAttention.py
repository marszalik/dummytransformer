import json
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
  pass