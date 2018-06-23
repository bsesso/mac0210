import numpy as np
import matplotlib.pyplot as plt
from splines import Spline
from splines import matrix_m2

class Spline_Interpolator:
  def __init__(self, xmin=0, xmax=1):
    self.xmin = xmin
    self.xmax = xmax

  def interpolate(self, measurements, n=-1, lmbda=10):
    if n == -1:
      self.n = self.calculate_n(measurements)
    self.lmbda = lmbda

    self.gammas = Gammas(self.n, self.xmin, self.xmax)
    # M, b = self.calculate_m_b(self.n, gammas, measurements)
    M = self.calculate_M(measurements)
    b = self.calculate_b(measurements)

    a = np.linalg.solve(M, 0.5 * b)
    final_curve = Spline(a, xmin, xmax)

    return final_curve

  def calculate_n(self, measurements):
    m = len(measurements)
    return max(m / 2, 10)

  def calculate_M(self, measurements):
    m = len(measurements)
    M = np.zeros((self.n, self.n))
    for j in range(m):
      M += self.gammas.get(j).dot(self.gammas.get(j).T)

    lmbda = 10
    M += self.lmbda * matrix_m2(self.n)

    return M

  def calculate_b(self, measurements):
    m = len(measurements)
    b = np.zeros((1, self.n))
    for j in range(m):
      b += measurements[j] * self.gammas.get(j).T
    b = 2 * b.T
    return b

class Gammas:
  def __init__(self, n_splines, xmin = 0, xmax = 1):
    self.splines = Spline(np.ones(n_splines), xmin, xmax)
    self.n = n_splines

    self.current = -1
    self.array = None

  def get(self, j):
    if j != self.current:
      self.array = np.array([[self.splines.beta_j(i, j) for i in range(self.n)]]).T
      self.current = j
    return self.array

def plot_curve(curve):
  # Plota sempre 100 pontos
  dt = (curve.x_max - curve.x_min) / 100
  # Adiciono 0.3 para ver no plot os lados além do interpolado mesmo
  t = np.arange(curve.x_min - 0.3, curve.x_max + 0.3, dt)  
  plt.plot(t, curve(t))
  plt.show()

if __name__ == '__main__':
  n_spl = 10
  xmin = 0
  xmax = 10
  msmnts = np.array([0, 1, 2, 3, 4, 3, 2, 5, 9, 6, 4, 3, 2, 1])

  ip = Spline_Interpolator(xmin, xmax)

  curve = ip.interpolate(msmnts)
  plot_curve(curve)
