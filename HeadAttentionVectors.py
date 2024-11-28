class HeadAttentionVectors:
  def __init__(self):
    self.q_matrices = []
    self.k_matrices = []
    self.v_matrices = []

  def append(self, q, k, v):
    self.q_matrices.append(q)
    self.k_matrices.append(k)
    self.v_matrices.append(v)
    
  def get_q_matrix(self, index):
    return self.q_matrices[index]
    
  def get_k_matrix(self, index):
    return self.k_matrices[index]
    
  def get_v_matrix(self, index):
    return self.v_matrices[index]
  
  def len(self):
    return len(self.q_matrices)
    
  def __len__(self):
    return self.len()