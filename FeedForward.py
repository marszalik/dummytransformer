import numpy as np
import json


class FeedForward:

  def __init__(self):
    """Inicjalizuje sieć i ładuje wagi z pliku JSON."""
    self.weights = self.load_weights("feedForwardWeights.json")
    self.W1 = np.array(self.weights['W1'])
    self.b1 = np.array(self.weights['b1'])
    self.W2 = np.array(self.weights['W2'])
    self.b2 = np.array(self.weights['b2'])

  def load_weights(self, weights_file):
    """Ładuje wagi i biasy z pliku JSON."""
    with open(weights_file, 'r') as file:
      return json.load(file)

  def relu(self, x):
    """Funkcja aktywacji ReLU."""
    result = np.maximum(0, x)
    return result

  def compute_hidden_layer(self, input_vectors):
    """Oblicza warstwę ukrytą."""
    weighted_sum = np.dot(input_vectors, self.W1)
    biased_sum = weighted_sum + self.b1
    after_activation = self.relu(biased_sum)
    return after_activation

  def compute_output_layer(self, hidden_layer):
    """Oblicza warstwę wyjściową."""
    weighted_sum = np.dot(hidden_layer, self.W2)
    biased_sum = weighted_sum + self.b2
    return biased_sum

  def run(self, input_vectors):
    """Przetwarza wektory wejściowe przez sieć FeedForward."""
    hidden_layer = self.compute_hidden_layer(input_vectors)
    output_layer = self.compute_output_layer(hidden_layer)
    return output_layer
