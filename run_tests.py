import unittest
import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

def run_all_tests():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\nAll unit tests passed successfully.")
        return 0
    else:
        print(f"\nUnit test failures detected: {len(result.failures)} failures, {len(result.errors)} errors.")
        return 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
