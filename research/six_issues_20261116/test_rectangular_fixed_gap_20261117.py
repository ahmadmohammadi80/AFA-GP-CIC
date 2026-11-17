"""Counterexample: rectangular robust reachability is NOT exact for a fixed Bernoulli parameter."""
import unittest
from functools import lru_cache

def rectangular_value(n, s, t):
    @lru_cache(None)
    def value(n, s, t):
        if n == 2 and s == 1:
            return 1.0
        if t == 0:
            return 0.0
        v1 = value(n + 1, s + 1, t - 1)
        v0 = value(n + 1, s, t - 1)
        return min(v0 + p * (v1 - v0) for p in (0.2, 0.8))
    return value(n, s, t)

class TestFixedVsRectangular(unittest.TestCase):
    def test_strict_gap(self):
        # Event: exactly one success in two draws. C is deterministic but
        # intentionally non-monotone; this is a scope/falsification test.
        self.assertAlmostEqual(rectangular_value(0, 0, 2), 0.2)
        fixed_min = min(2 * p * (1 - p) for p in (0.2, 0.8))
        self.assertAlmostEqual(fixed_min, 0.32)
        self.assertLess(rectangular_value(0, 0, 2), fixed_min)

if __name__ == "__main__":
    unittest.main()
