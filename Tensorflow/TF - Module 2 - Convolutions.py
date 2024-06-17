import numpy as np

h = [2, 1, 0] # image
x = [3, 4, 5] # kernel

y = np.convolve(x, h)
print("Compare with the following values from Python: y[0] = {0} ; y[1] = {1}; y[2] = {2}; y[3] = {3}; y[4] = {4}".format(y[0], y[1], y[2], y[3], y[4]))

# 3 methods to apply kernel on matrix: with padding full, with padding same, without padding valid

# 1) padding full
# with zero padding
[2  6]
 |  |
 V  V
 0 [1 2 5 4] # 2*0 + 1*6 = 6

  [2  6]
   |  |
   V  V
0 [1  2  5  4]  # 2*1 + 6*2 = 14

     [2  6]
      |  |
      V  V
0 [1  2  5  4]  # 2*2 + 6*5 = 34

        [2  6]
         |  |
         V  V
0 [1  2  5  4]  # 2*5 + 6*4 = 34

           [2  6]
            |  |
            V  V
0 [1  2  5  4] 0  # 2*4 + 6*0 = 8

# Result of convolution is Y = [6 14 34 34 8]
import numpy as np

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, "full")  #now, because of the zero padding, the final dimension of the array is bigger
 # print (y), Y = [6 14 34 34 8]

# 2) padding same - only add 0 to left
import numpy as np

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, "same")  # it is same as zero padding, but with returns an ouput with the same length as max of x or h
y # array([ 6, 14, 34, 34])


# 3) No padding
  [2  6]
   |  |
   V  V
  [1  2  5  4]  # 2*1 + 6*2 = 14

     [2  6]
      |  |
      V  V
  [1  2  5  4]  # 2*2 + 6*5 = 34

        [2  6]
         |  |
         V  V
  [1  2  5  4]  # 2*5 + 6*4 = 34

import numpy as np

x = [6, 2]
h = [1, 2, 5, 4]

y = np.convolve(x, h, "valid")   # valid returns output of length max(x, h) - min(x, h) + 1, this is to ensure that values outside of the boundary of
                                # h will not be used in the calculation of the convolution
                                # in the next example we will understand why we used the argument valid
y  # array([14, 34, 34])

# 2D CONVOLUTION
# image represented by a 3x3 matrix according to the function g = (-1 1).

from scipy import signal as sg

I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1, 1]] # kernel

print('Without zero padding \n')
print('{0} \n'.format(sg.convolve( I, g, 'valid')))
# The 'valid' argument states that the output consists only of those elements
# that do not rely on the zero-padding.

print('With zero padding \n')
print(sg.convolve( I, g))

# Without zero padding
[[248   4]
 [-28 236]
 [  2 -14]]

# With zero padding

[[-255  248    4    3]
 [-212  -28  236    4]
 [-218    2  -14  230]]

from scipy import signal as sg

I= [[255,   7,  3],
    [212, 240,  4],
    [218, 216, 230],]

g= [[-1,  1],
    [ 2,  3],]

print ('With zero padding \n')
print ('{0} \n'.format(sg.convolve( I, g, 'full')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix

print ('With zero padding_same_ \n')
print ('{0} \n'.format(sg.convolve( I, g, 'same')))
# The output is the full discrete linear convolution of the inputs.
# It will use zero to complete the input matrix


print ('Without zero padding \n')
print (sg.convolve( I, g, 'valid'))
# The 'valid' argument states that the output consists only of those elements
#that do not rely on the zero-padding.

With zero padding

[[-255  248    4    3]
 [ 298  751  263   13]
 [ 206 1118  714  242]
 [ 436 1086 1108  690]]

With zero padding_same_

[[-255  248    4]
 [ 298  751  263]
 [ 206 1118  714]]

Without zero padding

[[ 751  263]
 [1118  714]]

# CODING WITH TENSORFLOW
import tensorflow as tf

#Building graph

input = tf.Variable(tf.random_normal([1, 10, 10, 1]))
filter = tf.Variable(tf.random_normal([3, 3, 1, 1]))
op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
op2 = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')

#Initialization and session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    print("Input \n")
    print('{0} \n'.format(input.eval()))
    print("Filter/Kernel \n")
    print('{0} \n'.format(filter.eval()))
    print("Result/Feature Map with valid positions \n")
    result = sess.run(op)
    print(result)
    print('\n')
    print("Result/Feature Map with padding \n")
    result2 = sess.run(op2)
    print(result2)

# Input
#
# [[[[ 1.3705449 ]
#    [ 0.65081406]
#    [-0.69144183]
#    [ 1.9935501 ]
#    [-0.12477966]
#    [ 1.307198  ]
#    [-0.56663543]
#    [ 0.8269176 ]
#    [-0.12183445]
#    [-0.23320168]]
#
#   [[-0.30322197]
#    [ 0.33855906]
#    [-0.13547783]
#    [-0.7415282 ]
#    [ 1.3021045 ]
#    [ 0.02860406]
#    [ 0.0954376 ]
#    [-3.9259279 ]
#    [ 0.92202795]
#    [-1.2484583 ]]
#
#   [[-0.2135468 ]
#    [ 0.07540299]
#    [-1.1409128 ]
#    [-0.97709024]
#    [-2.8967917 ]
#    [-1.6480644 ]
#    [ 0.15203308]
#    [ 1.8455061 ]
#    [-1.2017169 ]
#    [-0.37989318]]
#
#   [[-1.8771582 ]
#    [ 1.0908089 ]
#    [-0.71658   ]
#    [ 1.8985531 ]
#    [-0.22495341]
#    [ 0.1782781 ]
#    [ 0.48158672]
#    [-0.9393532 ]
#    [-0.7382227 ]
#    [-0.5059646 ]]
#
#   [[ 0.08520918]
#    [-0.97549754]
#    [-0.33358237]
#    [-0.35564852]
#    [-0.39148617]
#    [-0.19270189]
#    [ 0.436088  ]
#    [-0.4779004 ]
#    [ 0.9386145 ]
#    [ 0.48607832]]
#
#   [[ 0.7380855 ]
#    [-0.5738704 ]
#    [-0.3594812 ]
#    [ 1.0149981 ]
#    [-0.7548175 ]
#    [-0.73427683]
#    [ 1.2483801 ]
#    [ 0.53949255]
#    [-0.8603831 ]
#    [-0.23389459]]
#
#   [[ 0.8608204 ]
#    [-0.7079926 ]
#    [ 0.8660076 ]
#    [ 0.29789358]
#    [ 1.5824294 ]
#    [-1.4165289 ]
#    [ 0.35778692]
#    [ 0.8827392 ]
#    [-1.4261429 ]
#    [ 0.05328877]]
#
#   [[-1.5178634 ]
#    [-1.326157  ]
#    [-0.3906895 ]
#    [ 1.7758785 ]
#    [-1.3053011 ]
#    [ 0.714717  ]
#    [ 0.434984  ]
#    [ 0.47254828]
#    [-1.5954795 ]
#    [-0.3365681 ]]
#
#   [[-1.9130216 ]
#    [-0.15082704]
#    [-1.4152524 ]
#    [ 1.1701833 ]
#    [-0.713099  ]
#    [ 1.2247672 ]
#    [-0.28592587]
#    [-2.0046353 ]
#    [-2.6599545 ]
#    [-0.3691852 ]]
#
#   [[-0.39576793]
#    [ 0.9514967 ]
#    [-1.0960482 ]
#    [-0.86800283]
#    [-1.9522138 ]
#    [ 1.164837  ]
#    [-0.6345684 ]
#    [-0.92108   ]
#    [ 0.58453727]
#    [ 0.39060572]]]]
#
# Filter/Kernel
#
# [[[[-0.60371417]]
#
#   [[-0.8416426 ]]
#
#   [[ 1.4008412 ]]]
#
#
#  [[[ 0.57224876]]
#
#   [[ 1.1018904 ]]
#
#   [[-2.6045487 ]]]
#
#
#  [[[ 0.04053949]]
#
#   [[-0.5168885 ]]
#
#   [[-0.13947836]]]]
#
# Result/Feature Map with valid positions
#
# [[[[ -1.6798748 ]
#    [  5.6865573 ]
#    [ -4.8583894 ]
#    [  3.3561764 ]
#    [ -0.5773285 ]
#    [ 10.790108  ]
#    [ -7.977493  ]
#    [  2.0466337 ]]
#
#   [[  2.1007993 ]
#    [  0.35154095]
#    [  7.366222  ]
#    [  0.10158841]
#    [ -4.71457   ]
#    [-11.290163  ]
#    [ 10.396733  ]
#    [  0.9806759 ]]
#
#   [[  1.0155444 ]
#    [ -5.3817983 ]
#    [ -0.05405053]
#    [  1.3083214 ]
#    [  2.185197  ]
#    [  6.3649654 ]
#    [ -2.031412  ]
#    [ -1.2404172 ]]
#
#   [[ -0.56922966]
#    [  2.625621  ]
#    [ -1.4774842 ]
#    [ -0.30638677]
#    [ -0.73697126]
#    [ -0.9641631 ]
#    [ -3.364228  ]
#    [  0.47361833]]
#
#   [[  1.3086562 ]
#    [ -3.5145266 ]
#    [  2.4913855 ]
#    [  1.3275336 ]
#    [ -2.7366462 ]
#    [ -1.7353858 ]
#    [  4.7606964 ]
#    [  0.9148317 ]]
#
#   [[ -2.330826  ]
#    [  1.7445633 ]
#    [ -5.7440257 ]
#    [  5.244465  ]
#    [  0.752269  ]
#    [ -2.8289294 ]
#    [  2.4748132 ]
#    [ -0.24352771]]
#
#   [[  0.17482594]
#    [ -5.1366944 ]
#    [  6.0133977 ]
#    [ -5.534382  ]
#    [ -0.9763716 ]
#    [  1.9251995 ]
#    [  3.3639555 ]
#    [  1.4761355 ]]
#
#   [[  3.5553997 ]
#    [ -0.35022298]
#    [ -0.07395241]
#    [ -1.4669888 ]
#    [  1.8893092 ]
#    [  5.9750557 ]
#    [  2.0289762 ]
#    [ -2.9244685 ]]]]
#
#
# Result/Feature Map with padding
#
# [[[[ -0.07537627]
#    [  3.1339188 ]
#    [ -5.394585  ]
#    [  2.3221717 ]
#    [ -3.1084416 ]
#    [  2.869504  ]
#    [ -1.530664  ]
#    [  2.808773  ]
#    [  0.4847317 ]
#    [  0.35600996]]
#
#   [[ -1.35787   ]
#    [ -1.6798748 ]
#    [  5.6865573 ]
#    [ -4.8583894 ]
#    [  3.3561764 ]
#    [ -0.5773285 ]
#    [ 10.790108  ]
#    [ -7.977493  ]
#    [  2.0466337 ]
#    [ -0.4305638 ]]
#
#   [[  1.1159134 ]
#    [  2.1007993 ]
#    [  0.35154095]
#    [  7.366222  ]
#    [  0.10158841]
#    [ -4.71457   ]
#    [-11.290163  ]
#    [ 10.396733  ]
#    [  0.9806759 ]
#    [ -0.3805672 ]]
#
#   [[ -4.5321126 ]
#    [  1.0155444 ]
#    [ -5.3817983 ]
#    [ -0.05405053]
#    [  1.3083214 ]
#    [  2.185197  ]
#    [  6.3649654 ]
#    [ -2.031412  ]
#    [ -1.2404172 ]
#    [ -0.14793411]]
#
#   [[  5.441103  ]
#    [ -0.56922966]
#    [  2.625621  ]
#    [ -1.4774842 ]
#    [ -0.30638677]
#    [ -0.73697126]
#    [ -0.9641631 ]
#    [ -3.364228  ]
#    [  0.47361833]
#    [  2.030261  ]]
#
#   [[  0.5235314 ]
#    [  1.3086562 ]
#    [ -3.5145266 ]
#    [  2.4913855 ]
#    [  1.3275336 ]
#    [ -2.7366462 ]
#    [ -1.7353858 ]
#    [  4.7606964 ]
#    [  0.9148317 ]
#    [ -1.811198  ]]
#
#   [[  2.336962  ]
#    [ -2.330826  ]
#    [  1.7445633 ]
#    [ -5.7440257 ]
#    [  5.244465  ]
#    [  0.752269  ]
#    [ -2.8289294 ]
#    [  2.4748132 ]
#    [ -0.24352771]
#    [  0.06817925]]
#
#   [[  1.0750887 ]
#    [  0.17482594]
#    [ -5.1366944 ]
#    [  6.0133977 ]
#    [ -5.534382  ]
#    [ -0.9763716 ]
#    [  1.9251995 ]
#    [  3.3639555 ]
#    [  1.4761355 ]
#    [ -0.3847453 ]]
#
#   [[ -2.223486  ]
#    [  3.5553997 ]
#    [ -0.35022298]
#    [ -0.07395241]
#    [ -1.4669888 ]
#    [  1.8893092 ]
#    [  5.9750557 ]
#    [  2.0289762 ]
#    [ -2.9244685 ]
#    [ -0.8606765 ]]
#
#   [[ -1.5155166 ]
#    [  2.975995  ]
#    [  4.5189576 ]
#    [  2.3715727 ]
#    [ -4.0722914 ]
#    [  0.81829   ]
#    [ -0.94058824]
#    [ -4.7668867 ]
#    [  2.0314455 ]
#    [  2.6814797 ]]]]

# CONVOLUTION ON APPLIED IMAGES
# download standard image
!wget --quiet https://ibm.box.com/shared/static/cn7yt7z10j8rx6um1v9seagpgmzzxnlz.jpg --output-document bird.jpg


#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('bird.jpg')  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white
                              # uses the ITU-R 601-2 Luma transform (there are several
                              # ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)
### Activating matplotlib for Ipython
%matplotlib inline

### Plot image

imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)

kernel = np.array([[ 0, 1, 0],
                   [ 1,-4, 1],
                   [ 0, 1, 0],])

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')

%matplotlib inline

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')

# you usually convert the pixels values to a range from 0 to 1. This process is called normalization.
type(grad)

grad_biases = np.absolute(grad) + 100

grad_biases[grad_biases > 255] = 255

%matplotlib inline

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad_biases), cmap='gray')


#  Original type: <PIL.Image.Image image mode=L size=1920x1440 at 0x7F540CACB7F0>
#
#
# After conversion to numerical representation:
#
#  array([[ 64,  71,  65, ...,  49,  47,  48],
#        [ 68,  71,  64, ...,  54,  52,  51],
#        [ 65,  69,  66, ...,  54,  50,  55],
#        ...,
#        [ 21,  24,  23, ..., 184, 170, 155],
#        [ 18,  21,  26, ..., 179, 166, 153],
#        [ 27,  22,  21, ..., 170, 159, 149]], dtype=uint8)
#
#  Input image converted to gray scale:
#
#
# GRADIENT MAGNITUDE - Feature map
# <matplotlib.image.AxesImage at 0x7f544f331dd8>

# example using a digit
# download standard image
!wget --quiet https://ibm.box.com/shared/static/vvm1b63uvuxq88vbw9znpwu5ol380mco.jpg --output-document num3.jpg
#Importing
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from PIL import Image

im = Image.open('num3.jpg')  # type here your image's name

image_gr = im.convert("L")    # convert("L") translate color images into black and white
                              # uses the ITU-R 601-2 Luma transform (there are several
                              # ways to convert an image to grey scale)
print("\n Original type: %r \n\n" % image_gr)

# convert image to a matrix with values from 0 to 255 (uint8)
arr = np.asarray(image_gr)
print("After conversion to numerical representation: \n\n %r" % arr)
### Activating matplotlib for Ipython
%matplotlib inline

### Plot image
fig, aux = plt.subplots(figsize=(10, 10))
imgplot = plt.imshow(arr)
imgplot.set_cmap('gray')  #you can experiment different colormaps (Greys,winter,autumn)
print("\n Input image converted to gray scale: \n")
plt.show(imgplot)
kernel = np.array([
                        [ 0, 1, 0],
                        [ 1,-4, 1],
                        [ 0, 1, 0],
                                     ])

grad = signal.convolve2d(arr, kernel, mode='same', boundary='symm')
%matplotlib inline

print('GRADIENT MAGNITUDE - Feature map')

fig, aux = plt.subplots(figsize=(10, 10))
aux.imshow(np.absolute(grad), cmap='gray')
