"""Train a perceptron for AND with bipolar inputs and targets."""
import math

AND_EXAMPLES = ((-1, -1, -1), (-1, 1, -1), (1, -1, -1), (1, 1, 1))


class Perceptron:
    def __init__(self, learning_rate=0.1):
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.learning_rate = learning_rate
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.epochs = 0

    def predict(self, x1, x2):
        if x1 not in (-1, 1) or x2 not in (-1, 1):
            raise ValueError("Inputs must be -1 or 1")
        total = self.weights[0] * x1 + self.weights[1] * x2 + self.bias
        return 1 if total >= 0 else -1

    def fit(self, examples=AND_EXAMPLES, max_epochs=100):
        if not isinstance(max_epochs, int) or max_epochs <= 0:
            raise ValueError("max_epochs must be a positive integer")
        rows = list(examples)
        if not rows or any(len(row) != 3 or any(value not in (-1, 1) for value in row) for row in rows):
            raise ValueError("Each example must contain two bipolar inputs and one bipolar target")
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for x1, x2, target in rows:
                prediction = self.predict(x1, x2)
                if prediction != target:
                    delta = self.learning_rate * (target - prediction)
                    self.weights[0] += delta * x1
                    self.weights[1] += delta * x2
                    self.bias += delta
                    errors += 1
            self.epochs = epoch
            if errors == 0:
                return self
        raise RuntimeError("The perceptron did not converge within max_epochs")


if __name__ == "__main__":
    model = Perceptron().fit()
    print("x1 x2 AND")
    for x1, x2, target in AND_EXAMPLES:
        print(x1, x2, model.predict(x1, x2))
    print("Converged in", model.epochs, "epochs")
