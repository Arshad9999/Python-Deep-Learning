import contextlib
import importlib.util
import io
import math
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load_lab():
    spec = importlib.util.spec_from_file_location("multiclass_lab", ROOT / "LAB 03/code.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MulticlassLabTests(unittest.TestCase):
    def test_import_has_no_output_or_training_side_effect(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            module = load_lab()
        self.assertEqual(stream.getvalue(), "")
        self.assertTrue(hasattr(module, "NeuralNetwork"))

    def test_probabilities_are_finite_and_normalized(self):
        model = load_lab().NeuralNetwork()
        probabilities = model.predict_proba([0.5, -0.25])
        self.assertEqual(len(probabilities), 3)
        self.assertTrue(all(math.isfinite(value) and 0 < value < 1 for value in probabilities))
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_backpropagation_reduces_loss_and_classifies_holdout(self):
        module = load_lab()
        train, test = module.stratified_split(module.synthetic_flowers())
        model = module.NeuralNetwork()
        initial_loss = model.loss(train)
        history = model.fit(train)
        self.assertLess(history[-1], initial_loss * 0.08)
        self.assertGreaterEqual(model.accuracy(train), 0.95)
        self.assertGreaterEqual(model.accuracy(test), 0.90)

    def test_seeded_training_is_repeatable(self):
        module = load_lab()
        train, test = module.stratified_split(module.synthetic_flowers())
        first = module.NeuralNetwork()
        second = module.NeuralNetwork()
        self.assertEqual(first.fit(train, epochs=20), second.fit(train, epochs=20))
        self.assertEqual([first.predict(row[0]) for row in test], [second.predict(row[0]) for row in test])

    def test_invalid_shapes_labels_and_training_parameters(self):
        module = load_lab()
        model = module.NeuralNetwork()
        for features in ([1], [1, 2, 3], [1, float("nan")], ["1", 2]):
            with self.subTest(features=features), self.assertRaises(ValueError):
                model.predict(features)
        for rows in ([], [([1, 2], -1)], [([1, 2], 3)], [([1], 0)], [(1, 0, 2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                model.fit(rows)
        for epochs in (0, -1, 1.5):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                model.fit([([1, 2], 0)], epochs=epochs)

    def test_example_command_runs(self):
        result = subprocess.run([sys.executable, str(ROOT / "LAB 03/code.py")], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Test accuracy:", result.stdout)
        self.assertIn("Probabilities for", result.stdout)


if __name__ == "__main__":
    unittest.main()
