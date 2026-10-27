# Lab 03 — multiclass backpropagation

This lab implements a one-hidden-layer classifier using only Python's standard library. It demonstrates deterministic initialization, a stratified train/test split, a `tanh` hidden layer, numerically stable softmax, multiclass cross-entropy, stochastic gradient descent and backpropagation through both trainable layers.

The bundled observations are explicitly **synthetic** three-class points. They make the mechanics reproducible without downloading a dataset or implying that a tiny exercise is an evaluated Iris model. `NeuralNetwork` accepts any finite numeric feature rows with integer labels, so a permitted real dataset can be supplied later with its source, license, preprocessing and evaluation protocol documented separately.

Run the example from the repository root:

```sh
python3 "LAB 03/code.py"
```

The example uses a deterministic holdout to verify that training generalizes beyond the rows used for weight updates. This is a teaching check, not a performance claim about real flower measurements. For a real study, decide the split before tuning, fit scaling on training data only, compare meaningful baselines and report uncertainty across appropriate repeated evaluations.
