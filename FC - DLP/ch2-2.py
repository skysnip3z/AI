# -*- coding: utf-8 -*-
"""

@author: Ali Hassan
"""
#%% Imports

import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

#%% Tensor ranks from 0 - 3+ (example is int32 can be dtype="float32/64")

# Rank 0 Tensor (0D, Scalar)
a = np.array(12, dtype="float32")
print(a)
print(a.ndim)

# Rank 1 Tensor (1D, Vector)
b = np.array([12, 3, 6 , 14, 7], dtype="float32") # Five dimensional vector not 5D tensor
print(b)
print(b.ndim)

# Rank 2 Tensor (2D, Matrix)
c = np.array([[5, 78, 2, 34, 0],
              [6, 79, 3, 35, 1],
              [7, 80, 4, 36, 2]], dtype="float32")
print(c)
print(c.ndim)

# Rank 3 Tensor (3D<=, Cube+)
d = np.array([[[5, 56, 1, 14, 9],
               [6, 41, 2, 75, 8],
               [7, 11, 4, 66, 6]],
              [[1, 98, 5, 24, 7],
               [2, 89, 7, 15, 3],
               [3, 70, 8, 96, 2]],
              [[4, 68, 1, 64, 5],
               [5, 59, 2, 15, 3],
               [6, 40, 7, 96, 1]],
              [[7, 38, 8, 84, 9],
               [8, 29, 9, 55, 5],
               [9, 10, 1, 46, 4]]], dtype="float32")
print(d)
print(d.ndim)

#%% MNIST - Tensor Axes, Shape, and Datatype - Load Data

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%% MNIST - Tensor Axes, Shape, and Datatype - Explore

print(train_images.ndim) # Number of Axes
print(train_images.shape) # Shape of the data
print(train_images.dtype) # Datatype of the data

# Plotting an example from the data
digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# Confirming plot was correct

print(train_labels[4])

#%% Manipulating tensors

slice_a = train_images[10:100] # 10-99!100 = 90
#print(slice_a.shape)
digit_a = slice_a[0]
plt.imshow(digit_a, cmap=plt.cm.binary)
plt.show()

slice_b = train_images[10:100, :, :] # same as above, more detailed
#print(slice_b.shape)
digit_b = slice_b[0]
plt.imshow(digit_b, cmap=plt.cm.binary)
plt.show()

slice_c = train_images[10:100, 0:28, 0:28] # same as above, more detailed
#print(slice_c.shape)
digit_c = slice_c[0]
plt.imshow(digit_c, cmap=plt.cm.binary)
plt.show()

slice_d = train_images[:, 14:, 14:] # bottom right corner of image 14x14
#print(slice_d.shape)
digit_d = slice_d[10] # equivalent to index 0 from previous examples
plt.imshow(digit_d, cmap=plt.cm.binary)
plt.show()

slice_e = train_images[:, 7:-7, 7:-7] # 14x14 centred in middle
#print(slice_e.shape)
digit_e = slice_e[10] # same as previous
plt.imshow(digit_e, cmap=plt.cm.binary)
plt.show()

slice_f = train_images[:, :14, :14] #  Top left corner of image 14x14
#print(slice_f.shape)
digit_f = slice_f[10] # equivalent to index 0 from previous examples
plt.imshow(digit_f, cmap=plt.cm.binary)
plt.show()
# y,x
slice_g = train_images[:, :14, 14:] #  Top right corner of image 14x14
#print(slice_g.shape)
digit_g = slice_g[10] # equivalent to index 0 from previous examples
plt.imshow(digit_g, cmap=plt.cm.binary)
plt.show()

slice_h = train_images[:, 14:, :14] # Bottom left corner of image 14x14
#print(slice_h.shape)
digit_h = slice_h[10] # equivalent to index 0 from previous examples
plt.imshow(digit_h, cmap=plt.cm.binary)
plt.show()

# 14:x, 14:x - Bottom Right
# 14:x, x:14 - Bottom Left
# x:14, 14:x - Top Right
# x:14, x:14 - Top Left
# 0 <-- 14 --> 28

#%% Data batches - Axis 0 is usually samples axis/dimension

# Data is processed in smaller batches
batch_zero = train_images[:128]
batch_one = train_images[128:256]
n = 1 # The nth batch - Is same as batch_one but to n
batch_n = train_images[128 * n : 128 * (n + 1)] 





 



