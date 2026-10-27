import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    spec = importlib.util.spec_from_file_location("logic_lab", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LogicLabTests(unittest.TestCase):
    def test_xor_truth_table(self):
        module = load("LAB 01/XORfunctionUsingMcCulloch-PittsNeuron.py")
        for x1, x2, expected in ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)):
            with self.subTest(inputs=(x1, x2)):
                self.assertEqual(module.xor(x1, x2), expected)

    def test_xor_rejects_nonbinary_inputs(self):
        module = load("LAB 01/XORfunctionUsingMcCulloch-PittsNeuron.py")
        for x1, x2 in ((-1, 1), (0, 2), (0, "1")):
            with self.subTest(inputs=(x1, x2)), self.assertRaises(ValueError):
                module.xor(x1, x2)

    def test_perceptron_learns_all_bipolar_and_cases(self):
        module = load("LAB 02/code.py")
        for rate in (0.01, 0.1, 1.0):
            model = module.Perceptron(rate).fit()
            for x1, x2, target in module.AND_EXAMPLES:
                with self.subTest(rate=rate, inputs=(x1, x2)):
                    self.assertEqual(model.predict(x1, x2), target)
            self.assertLessEqual(model.epochs, 100)

    def test_perceptron_reports_nonseparable_xor(self):
        module = load("LAB 02/code.py")
        examples = ((-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1))
        with self.assertRaises(RuntimeError):
            module.Perceptron().fit(examples, max_epochs=20)

    def test_invalid_training_parameters_and_data(self):
        module = load("LAB 02/code.py")
        for rate in (0, -1, float("nan"), float("inf")):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                module.Perceptron(rate)
        for rows in ([], [(0, 1, 1)], [(1, 1)], [(1, 1, 0)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                module.Perceptron().fit(rows)
        for epochs in (0, -1, 1.5):
            with self.subTest(epochs=epochs), self.assertRaises(ValueError):
                module.Perceptron().fit(max_epochs=epochs)

    def test_importing_labs_has_no_training_or_output_side_effect(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            load("LAB 01/XORfunctionUsingMcCulloch-PittsNeuron.py")
            load("LAB 02/code.py")
        self.assertEqual(stream.getvalue(), "")

    def test_both_documented_commands_run(self):
        for path in ("LAB 01/XORfunctionUsingMcCulloch-PittsNeuron.py", "LAB 02/code.py"):
            with self.subTest(path=path):
                result = subprocess.run([sys.executable, str(ROOT / path)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("1 1 1" if "LAB 02" in path else "1 1 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
