import numpy as np

def initialise_input(N, d):
  '''
  N: Number of vectors
  d: dimension of vectors
  '''
  np.random.seed(0)
  U = np.random.randn(N, d)
  M1 = np.abs(np.random.randn(d, d))
  M2 = np.abs(np.random.randn(d, d))

  return U, M1, M2


def solve(N,d):

  '''
  Enter your code here for steps 1 to 6
  '''
  U, M1, M2 = initialise_input(N,d)

  #step 1
  X = np.matmul(U, M1)
  Y = np.matmul(U, M2)
  

  #step 2
  iRow = np.arange(1, N+1).reshape(-1, 1)
  X_hat = X + iRow
  

  #step 3
  Z = np.matmul(X_hat, np.transpose(Y))
  i, j = np.indices((N,N)) 

  mask = (i == j)|((i + j)% 2 == 0)
  Zsparse = np.where(mask, Z, 0)
  
  #step 4
  z_exp = np.exp(Zsparse)
  Z_hat = z_exp / np.sum(z_exp, axis = 1, keepdims = True)
  

  #step 5
  max_indices = np.argmax(Z_hat, axis = 1)
  return max_indices


solve()
# N = 10
# d = 8

# solve(N,d)
