# Import the Training Data

# This is for downloading training dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/training_data_pytorch.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_PYTORCH
!tar -xzf  /resources/data/training_data_pytorch.tar.gz -C /resources/data

# Import the Validation Data

# This is for downloading validation dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/validation_data_pytorch.tar.gz https://cocl.us/DL0320EN_VALID_TAR_PYTORCH
!tar -xzf  /resources/data/validation_data_pytorch.tar.gz -C /resources/data

# Import the Test Data

# This is for downloading test dataset. If you already have the dataset, you can comment it out in order to avoid the second time downloading.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

!wget --quiet -O /resources/data/test_data_pytorch.tar.gz https://cocl.us/DL0320EN_TEST_TAR_PYTORCH
!tar -xzf /resources/data/test_data_pytorch.tar.gz -C /resources/data --exclude '.*'


# Questions

# Import the libraries for plotting images

import matplotlib.pyplot as plt
from PIL import Image

# Question 1.1
# Load image 1 from the training data

train_dir = "/resources/data/training_data_pytorch"
name = "/0.jpeg"
Input = train_dir + name # Input= "/resources/data/training_data_pytorch/0.jpeg"

img = Image.open(Input)

plt.imshow(img)
plt.show()

# Question 1.2
# Load image 53 from the training data


train_dir = "/resources/data/training_data_pytorch"
name = "/52.jpeg"
Input = train_dir + name

img = Image.open(Input)

plt.imshow(img)
plt.show()

# Question 1.3
# Load and plot sample 1 (0.jpeg) from the validation data

validation_dir = "/resources/data/validation_data_pytorch"

name = "/0.jpeg"
Input = validation_dir + name

img = Image.open(Input)

plt.imshow(img)
plt.show()

# Question 1.4
# Load and plot sample 35 (36.jpeg) from the validation data.

validation_dir = "/resources/data/validation_data_pytorch"

name = "/0.jpeg"
Input = validation_dir + name

img = Image.open(Input)

plt.imshow(img)
plt.show()
