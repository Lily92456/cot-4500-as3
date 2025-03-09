import unittest
import numpy as np
from your_module import f, euler_method, runge_kutta_method  # Replace 'your_module' with the actual filename

class TestODESolvers(unittest.TestCase):
    
    def setUp(self):
        self.t0 = 0
        self.y0 = 1
        self.t_end = 2
        self.n = 10
        self.tolerance = 1e-2  # Tolerance for numerical solutions
    
    def test_euler_method(self):
        t_values, y_values = euler_method(f, self.t0, self.y0, self.t_end, self.n)
        expected_value = 0.3967  # Approximate expected value for y(2) using Euler's method
        self.assertAlmostEqual(y_values[-1], expected_value, delta=self.tolerance)
    
    def test_runge_kutta_method(self):
        t_values, y_values = runge_kutta_method(f, self.t0, self.y0, self.t_end, self.n)
        expected_value = 0.5042  # Approximate expected value for y(2) using Runge-Kutta method
        self.assertAlmostEqual(y_values[-1], expected_value, delta=self.tolerance)

if __name__ == "__main__":
    unittest.main()