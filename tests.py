import unittest
from main import add   # assuming file name is add.py
class TestAddFunction(unittest.TestCase):

    def test_positive_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_zero_and_number(self):
        self.assertEqual(add(0, 5), 5)

    def test_negative_numbers(self):
        self.assertEqual(add(-2, -3), -5)

    def test_positive_and_negative(self):
        self.assertEqual(add(5, -3), 2)

    def test_zeros(self):
        self.assertEqual(add(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
