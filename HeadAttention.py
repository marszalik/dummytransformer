class HeadAttention:
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
  pass