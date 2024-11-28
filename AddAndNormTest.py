import numpy as np
from AddAndNorm import AddAndNorm

residual_tensor = np.array([[1.0, 2.0, 3.0], [5.0, 5.0,5.0]])  # Oryginalne wejście
attention_output = np.array([[0.5, -1.0, 1.5], [1.0, 1.0,2.0]])  # Wynik z warstwy uwagi

#residual_tensor = np.array([1.0, 5.0, 1.0])  # Oryginalne wejście
#attention_output = np.array([0, 0, 0])  # Wynik z warstwy uwagi

# Inicjalizacja Add & Norm
add_norm = AddAndNorm()

# Obliczenie Add & Norm
output = add_norm.add_and_norm(attention_output, residual_tensor)
print("Wynik Add & Norm wiele:\n", output)

#--------------------------------

residual_tensor2 = np.array([1.0, 5.0, 1.0])  # Oryginalne wejście
attention_output2 = np.array([0, 0, 0])  # Wynik z warstwy uwagi

# Inicjalizacja Add & Norm
add_norm2 = AddAndNorm()

# Obliczenie Add & Norm
output2 = add_norm2.add_and_norm(attention_output2, residual_tensor2)
# Wyświetlenie wyniku
print("Wynik Add & Norm jeden:\n", output2)
