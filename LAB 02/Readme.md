# AND with a bipolar perceptron

Run `python3 "LAB 02/code.py"` from the repository root. Python 3 is the only requirement.

Inputs and targets use -1 and 1. The output is 1 only when both inputs are 1. `Perceptron.fit()` starts with zero weights, updates misclassified examples, and stops when an entire epoch has no errors. It raises an error rather than reporting success if the epoch limit is reached.

For a misclassification, delta = learning_rate × (target − prediction), each weight receives delta × input, and the bias receives delta. Prediction uses the sign of the weighted sum, with a zero sum mapped to 1.

All four possible inputs are used to verify this small logical function. They are not a held-out evaluation of generalization to a larger dataset. A single perceptron cannot solve XOR; the tests verify that the epoch limit catches that case.

The earlier implementation used binary targets despite the bipolar exercise title and updated one weight from the wrong previous value. Its original source remains in Git history.
