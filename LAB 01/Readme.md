# XOR with McCulloch–Pitts neurons

Run `python3 "LAB 01/XORfunctionUsingMcCulloch-PittsNeuron.py"` from the repository root. Python 3 is the only requirement.

| x1 | x2 | XOR |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The first layer computes OR with weights (1, 1) and threshold 1, and NAND with weights (-1, -1) and threshold -1. The output layer computes AND with weights (1, 1) and threshold 2. Each neuron fires when its weighted sum is at least its threshold.

This is a fixed logical network, so there is no fitting step or train/test accuracy claim. A single threshold neuron cannot implement XOR.

The earlier file mislabeled an AND table as XOR and used an incomplete training routine. Its original source remains available in Git history.
