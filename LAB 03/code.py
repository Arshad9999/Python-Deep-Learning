"""Dependency-free multiclass neural network trained by backpropagation.

Bundled observations are synthetic so this lab runs without downloads. The
model accepts any finite numeric feature matrix and integer class labels.
"""

from __future__ import annotations

import math
import random


def synthetic_flowers(seed=7, points_per_class=20):
    """Return a deterministic three-class, two-feature teaching dataset."""
    if not isinstance(points_per_class, int) or points_per_class < 2:
        raise ValueError("points_per_class must be an integer of at least 2")
    rng = random.Random(seed)
    centers = ((-1.8, -1.4), (1.8, -1.2), (0.0, 1.9))
    rows = []
    for label, (center_x, center_y) in enumerate(centers):
        for _ in range(points_per_class):
            rows.append(([rng.gauss(center_x, 0.28), rng.gauss(center_y, 0.28)], label))
    rng.shuffle(rows)
    return rows


class NeuralNetwork:
    """One-hidden-layer classifier using tanh, softmax and SGD."""

    def __init__(self, input_size=2, hidden_size=6, classes=3, learning_rate=0.08, seed=11):
        for name, value in (("input_size", input_size), ("hidden_size", hidden_size), ("classes", classes)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if classes < 2:
            raise ValueError("classes must be at least 2")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.classes = classes
        self.learning_rate = learning_rate
        rng = random.Random(seed)
        input_scale = math.sqrt(1 / input_size)
        hidden_scale = math.sqrt(1 / hidden_size)
        self.hidden_weights = [[rng.uniform(-input_scale, input_scale) for _ in range(input_size)] for _ in range(hidden_size)]
        self.hidden_bias = [0.0] * hidden_size
        self.output_weights = [[rng.uniform(-hidden_scale, hidden_scale) for _ in range(hidden_size)] for _ in range(classes)]
        self.output_bias = [0.0] * classes

    @staticmethod
    def _softmax(values):
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def _validate_features(self, features):
        try:
            values = list(features)
        except TypeError as error:
            raise ValueError("Features must be an iterable of finite numbers") from error
        if len(values) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("Features must be finite numbers")
        return [float(value) for value in values]

    def forward(self, features):
        values = self._validate_features(features)
        hidden = [math.tanh(sum(weight * value for weight, value in zip(row, values)) + bias) for row, bias in zip(self.hidden_weights, self.hidden_bias)]
        logits = [sum(weight * value for weight, value in zip(row, hidden)) + bias for row, bias in zip(self.output_weights, self.output_bias)]
        return hidden, self._softmax(logits)

    def predict_proba(self, features):
        return self.forward(features)[1]

    def predict(self, features):
        probabilities = self.predict_proba(features)
        return max(range(self.classes), key=probabilities.__getitem__)

    def _validate_rows(self, rows):
        checked = []
        for row in rows:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("Each row must be a (features, label) pair")
            features, label = row
            if not isinstance(label, int) or label not in range(self.classes):
                raise ValueError(f"Labels must be integers from 0 to {self.classes - 1}")
            checked.append((self._validate_features(features), label))
        if not checked:
            raise ValueError("At least one training row is required")
        return checked

    def loss(self, rows):
        checked = self._validate_rows(rows)
        return -sum(math.log(max(self.predict_proba(features)[label], 1e-15)) for features, label in checked) / len(checked)

    def fit(self, rows, epochs=250, shuffle_seed=13):
        checked = self._validate_rows(rows)
        if not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be a positive integer")
        rng = random.Random(shuffle_seed)
        history = []
        for _ in range(epochs):
            rng.shuffle(checked)
            for features, label in checked:
                hidden, probabilities = self.forward(features)
                output_delta = probabilities[:]
                output_delta[label] -= 1.0
                hidden_delta = [
                    (1 - hidden[index] ** 2)
                    * sum(self.output_weights[class_index][index] * output_delta[class_index] for class_index in range(self.classes))
                    for index in range(self.hidden_size)
                ]
                for class_index in range(self.classes):
                    for hidden_index in range(self.hidden_size):
                        self.output_weights[class_index][hidden_index] -= self.learning_rate * output_delta[class_index] * hidden[hidden_index]
                    self.output_bias[class_index] -= self.learning_rate * output_delta[class_index]
                for hidden_index in range(self.hidden_size):
                    for feature_index in range(self.input_size):
                        self.hidden_weights[hidden_index][feature_index] -= self.learning_rate * hidden_delta[hidden_index] * features[feature_index]
                    self.hidden_bias[hidden_index] -= self.learning_rate * hidden_delta[hidden_index]
            history.append(self.loss(checked))
        return history

    def accuracy(self, rows):
        checked = self._validate_rows(rows)
        return sum(self.predict(features) == label for features, label in checked) / len(checked)


def stratified_split(rows, test_per_class=4):
    grouped = {}
    for features, label in rows:
        grouped.setdefault(label, []).append((features, label))
    if not isinstance(test_per_class, int) or test_per_class < 1:
        raise ValueError("test_per_class must be a positive integer")
    if any(len(group) <= test_per_class for group in grouped.values()):
        raise ValueError("Every class needs more observations than test_per_class")
    train, test = [], []
    for label in sorted(grouped):
        train.extend(grouped[label][:-test_per_class])
        test.extend(grouped[label][-test_per_class:])
    return train, test


if __name__ == "__main__":
    training_rows, test_rows = stratified_split(synthetic_flowers())
    network = NeuralNetwork()
    losses = network.fit(training_rows)
    print(f"Initial/final loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"Training accuracy: {network.accuracy(training_rows):.1%}")
    print(f"Test accuracy: {network.accuracy(test_rows):.1%}")
    print("Probabilities for [0.0, 1.8]:", [round(value, 4) for value in network.predict_proba([0.0, 1.8])])
