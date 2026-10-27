"""Binary XOR using three fixed McCulloch–Pitts threshold neurons."""


def xor(x1, x2):
    """Return XOR for binary inputs using OR, NAND and AND neurons.

    A single threshold neuron cannot represent XOR. The first layer computes
    OR and NAND; the final neuron combines those two outputs with AND.
    """
    if x1 not in (0, 1) or x2 not in (0, 1):
        raise ValueError("XOR inputs must be 0 or 1")
    hidden_or = int(x1 + x2 >= 1)
    hidden_nand = int(-x1 - x2 >= -1)
    return int(hidden_or + hidden_nand >= 2)


if __name__ == "__main__":
    print("x1 x2 XOR")
    for x1, x2 in ((0, 0), (0, 1), (1, 0), (1, 1)):
        print(x1, x2, xor(x1, x2))
