import numpy as np
import matplotlib.pyplot as plt
from splines import Spline

def interpolate(n_splines, measurements, xmin = 0, xmax = 1):
  splines = generate_basic_splines(n_splines, xmin, xmax)  
  M, b = calculate_m_b(n_splines, splines, measurements)
  a = np.linalg.solve(M, 0.5 * b)
  final_curve = Spline(a, xmin, xmax)
  plot_curve(final_curve)

def generate_basic_splines(n, xmin, xmax):
  splines = []

  ## Cria cada spline 
  for i in range(n):
    a = np.array([0] * n)
    a[i] = 1

    spl = Spline(a, xmin, xmax)
    splines.append(spl)
  return splines

def calculate_m_b(n, splines, measurements):
  m = len(measurements)
  M = np.zeros((n, n))
  b = np.zeros((1, n))

  ## Calculate m = sum of gamma * gammaT && b = 2 * sum measurementsj * gammajT
  for j in range(m):
    gamma = np.array([[spl(j) for spl in splines]]).T
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
  n_spl = 8
  xmax = 10
  msmnts = np.array([0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 3, 2, 1])
  interpolate(n_splines=n_spl, measurements=msmnts, xmax=xmax)
