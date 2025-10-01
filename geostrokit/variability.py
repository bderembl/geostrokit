#!/usr/bin/env python

import numpy as np

def iterative_mean(it, psi_me, psi):
  '''
  Update the running mean of a field using an online algorithm.

  This function is an elementary building block for computing the mean 
  of a sequence of fields (e.g., arrays, tensors) in an iterative way. 
  It is particularly useful in contexts where loading or generating the 
  full dataset at once is expensive in terms of memory or computation.

  Parameters
  ----------

  it : int
      Iteration number (starting from 1). Corresponds to the number of samples
      seen so far, including the current one.
  psi_me : object
      Current mean (same shape and type as psi). Can be a NumPy array, tensor, etc.
  psi : object
      New field (e.g., a NumPy array or tensor) to incorporate into the mean.


  Returns
  -------

  psi_me : object
      Updated mean including the new field.


  Notes
  -----

  This implements the online (incremental) mean update formula:
      new_mean = old_mean + (new_sample - old_mean) / it


  Example
  -------

  >>> import numpy as np
  >>> psi = [np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([3.0, 6.0])]
  >>> psi_me = np.zeros_like(fields[0])
  >>> for i, psi_i in enumerate(psi, 1):
  ...     psi_me = iterative_mean(i, psi_me, psi_i)
  >>> print(psi_me)
  [2. 4.]

  '''

  return psi_me + (psi - psi_me)/it


def iterative_variance(it, psi_me, psi_var, psi):
  '''
  Update the running mean and variance of a field using Welford's online algorithm.

  This function is an elementary building block for computing the variance of a sequence 
  of fields (e.g., arrays, tensors) in an iterative fashion. It is especially useful 
  when loading or storing all data samples at once is expensive or impractical.

  Based on:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm


  Parameters
  ----------

  it : int
      Iteration number (starting from 1). Represents the total number of samples 
      seen so far, including the current one.
  psi_me : object
      Current mean estimate. Should be the same shape and type as `psi`.
  psi_var : object
      Current variance estimate. Same shape and type as `psi`.
  psi : object
      New sample (e.g., a NumPy array or tensor) to incorporate into the running statistics.


  Returns
  -------

  psi_me : object
      Updated mean including the new sample.
  psi_var : object
      Updated (biased) variance including the new sample.


  Notes
  -----

  This function implements Welford's algorithm for online variance:
      delta = x - mean
      mean = mean + delta / n
      delta2 = x - mean
      var = ((n - 1) * var + delta * delta2) / n


  Example
  -------

  >>> import numpy as np
  >>> psi = [np.array([1.0, 2.0]), np.array([2.0, 4.0]), np.array([3.0, 6.0])]
  >>> psi_me = np.zeros_like(psi[0])
  >>> psi_var = np.zeros_like(psi[0])
  >>> for i, psi_i in enumerate(fields, 1):
  ...     psi_me, psi_var = iterative_variance(i, psi_me, psi_var, psi_i)
  >>> print("Mean:", psi_me)
  >>> print("Variance:", psi_var)
  Mean: [2. 4.]
  Variance: [0.66666667 2.66666667]
  '''

  delta = psi - psi_me
  psi_me = psi_me + delta/it
  delta2 = psi - psi_me
  psi_var = (psi_var*(it-1) + delta*delta2)/it

  return psi_me, psi_var
