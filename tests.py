import unittest
import subprocess
import sys

class TestHelloWorld(unittest.TestCase):

    def test_direct_execution(self):
        result = subprocess.run(
            [sys.executable, "filename.py"],
            capture_output=True,
            text=True
        )
        self.assertIn("Hello, World!", result.stdout)
        self.assertIn("This is the main module.", result.stdout)

    def test_import_execution(self):
        result = subprocess.run(
            [sys.executable, "-c", "import filename"],
            capture_output=True,
            text=True
        )
        self.assertIn("Hello, World!", result.stdout)
        self.assertNotIn("This is the main module.", result.stdout)

if __name__ == "__main__":
    unittest.main()
