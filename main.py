import numpy as np
import matplotlib.pyplot as plt
from splines import Spline

def interpolate(n_splines, measurements, xmin = 0, xmax = 1):  
  M, b = calculate_m_b(n_splines, measurements)
  a = np.linalg.solve(M, 0.5 * b)
  final_curve = Spline(a, xmin, xmax)
  plot_curve(final_curve)

def calculate_m_b(n, measurements):
  m = len(measurements)
  M = np.zeros((n, n))
  b = np.zeros((1, n))
  splines = Spline(np.ones(n), 0, 10)

  ## Calculate m = sum of gamma * gammaT && b = 2 * sum measurementsj * gammajT
  for j in range(m):
    gamma = np.array([[splines.beta_j(i, j) for i in range(n)]]).T
    b += measurements[j] * gamma.T
    M += gamma.dot(gamma.T)
  b = 2 * b.T
  return (M, b)

def plot_curve(curve):
  # Plota sempre 100 pontos
  dt = (curve.x_max - curve.x_min) / 100
  # Adiciono 0.3 para ver no plot os lados além do interpolado mesmo
  t = np.arange(curve.x_min - 0.3, curve.x_max + 0.3, dt)  
  plt.plot(t, curve(t))
  plt.show()

if __name__ == '__main__':
  n_spl = 10
  xmax = 10
  msmnts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
  interpolate(n_splines=n_spl, measurements=msmnts, xmax=xmax)
