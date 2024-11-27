from FeedForward import FeedForward
import pprint
import numpy as np

# Przykładowe dane wejściowe
attention_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

# Inicjalizacja klasy FeedForward z wagami z pliku JSON
feed_forward = FeedForward()

# Przetwarzanie danych przez sieć
new_vectors = feed_forward.run(attention_vectors)

print("Wektory po feedforwardzie:")
pprint.pprint(new_vectors)
