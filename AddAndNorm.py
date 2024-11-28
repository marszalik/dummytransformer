import numpy as np


class AddAndNorm:

    def __init__(self, d_model):
        """
        Inicjalizacja klasy AddAndNorm.

        Args:
            d_model (int): Rozmiar modelu (długość wektora wejściowego).
        """
        self.d_model = d_model
        self.gamma = np.ones(d_model)  # Skalar dla LayerNorm
        self.beta = np.zeros(d_model)  # Przesunięcie dla LayerNorm

    def layer_norm(self, x):
        """
        Warstwa normalizacji (LayerNorm).

        Args:
            x (np.ndarray): Wejściowy wektor lub macierz.

        Returns:
            np.ndarray: Znormalizowany wektor lub macierz.
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        print(mean)
        variance = np.var(x, axis=-1, keepdims=True)
        print(variance)
        normalized = (x - mean) / np.sqrt(variance + 1e-6)
        return self.gamma * normalized + self.beta

    def add_and_norm(self, input_tensor, residual_tensor):
        """
        Operacja Add & Norm.

        Args:
            input_tensor (np.ndarray): Wejściowy tensor z warstwy uwagi (attention output).
            residual_tensor (np.ndarray): Tensor wejściowy oryginalnego słowa (residual connection).

        Returns:
            np.ndarray: Wynik Add & Norm.
        """
        # Dodawanie residuala
        added = input_tensor + residual_tensor
        print(added)
        # Normalizacja
        normalized = self.layer_norm(added)
        return normalized

    def add_and_norm_batch(self, input_tensors, residual_tensors):
        pass
