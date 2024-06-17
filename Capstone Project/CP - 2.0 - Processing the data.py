# You can comment the code in this box out if you already have the dataset.
# Step 1: Ctrl + A : Select all
# Step 2: Ctrl + / : Comment out all; if everything selected has been comment out alreaday, then uncomment all

# Download Training Dataset
!wget --quiet -O /resources/data/training_data_pytorch.tar.gz https://cocl.us/DL0320EN_TRAIN_TAR_PYTORCH
!tar -xzf  /resources/data/training_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# Download Validation Dataset
!wget --quiet -O /resources/data/validation_data_pytorch.tar.gz https://cocl.us/DL0320EN_VALID_TAR_PYTORCH
!tar -xzf  /resources/data/validation_data_pytorch.tar.gz -C /resources/data --exclude '.*'

# PyTorch Modules you need for this lab

from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms

# Other non-PyTorch Modules

from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image

# 1) Load CSV

# ) a. Training Data CSV
# train_csv_file contains the URL that contains the CSV file we needed

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'

# Read CSV file from the URL and print out the first five samples

train_data_name = pd.read_csv(train_csv_file) # convert CSV file to dataframe with read_csv()
train_data_name.head()

# The first column of the dataframe corresponds to the sample number. The second column is the denomination.
# The third column is the file name, and the final column is the class variable.
# The correspondence between the class variable and each denomination is as follows.
# Five Euros have y equal 0; ten Euros have y equals 1; twenty Euros have y equals 2 and so on.

# Get the value on location row 0, column 2 (Notice that index starts at 0.)

print('File name:', train_data_name.iloc[0, 2]) # The first argument corresponds to the sample number, and the second input corresponds to the column index.

# Get the value on location row 0, column 3 (Notice that index starts at 0.)

print('y:', train_data_name.iloc[0, 3])

# Print out the file name and the class number of the element on row 1 (the second row)

print('File name:', train_data_name.iloc[1, 2])
print('y:', train_data_name.iloc[1, 3])

# Print out the total number of rows in traing dataset
# You can obtain the number of rows using the following lines of code. This will correspond the data attribute len.
print('The number of rows: ', train_data_name.shape[0])

# b. Validation

# The url for getting csv file for validation dataset

validation_csv_file='https://cocl.us/DL0320EN_VALID_CSV'

validation_data_name = pd.read_csv(validation_csv_file)
validation_data_name.head()

# Load the 11th sample image name and class label
print("The file name: ", validation_data_name.iloc[10, 2])
print("The class label: ", validation_data_name.iloc[10, 3])

# 2) Load Image

# a. Training Images

# Save the image folderpath in a variable

train_data_dir = '/resources/data/training_data_pytorch/'

# Print the file name on the second row

train_data_name.iloc[1, 2]

# Combine the directory path with file name

train_image_name = train_data_dir + train_data_name.iloc[1, 2]

# Plot the second training image

image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

# Plot the 20th image

train_image_name = train_data_dir + train_data_name.iloc[19, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

# b. Validation Images

# Save the image folderpath in a variable

validation_data_dir='/resources/data/validation_data_pytorch/'

validation_data_name = pd.read_csv(validation_csv_file)
validation_image_name = validation_data_dir + validation_data_name.iloc[1, 2]
image = Image.open(validation_image_name)
plt.imshow(image)
plt.show()

# 3) Create a Dataset Class

# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        # Image directory
        self.data_dir = data_dir

        # The transform is goint to be used on image
        self.transform = transform

        # Load the CSV file contians image info
        self.data_name = pd.read_csv(csv_file)

        # Number of images in dataset
        self.len = self.data_name.shape[0]

        # Get the length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        # Image file path
        img_name = self.data_dir + self.data_name.iloc[idx, 2]

        # Open image file
        image = Image.open(img_name)

        # The class label for the image
        y = self.data_name.iloc[idx, 3]

        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y


# Test Transform

# Create the transform compose

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])

# Create a test_normalization dataset using composed as transform

test_normalization = Dataset(csv_file=train_csv_file
                        , data_dir='/resources/data/training_data_pytorch/'
                        , transform = composed)
# Print mean and std

print("Mean: ", test_normalization[0][0].mean(dim = 1).mean(dim = 1))
print("Std:", test_normalization[0][0].std(dim = 1).std(dim = 1))
# Mean:  tensor([0.4090, 0.6965, 0.9854])
# Std: tensor([0.2493, 0.2633, 0.2477])



# QUESTIONS

# Create the dataset objects

train_dataset = Dataset(csv_file=train_csv_file
                        , data_dir='/resources/data/training_data_pytorch/')
validation_data = Dataset(csv_file=validation_csv_file
                          , data_dir='/resources/data/validation_data_pytorch/')

# Question 2.1

# print out samples = [53, 23, 10] from training data

# 1) Load CSV

# ) a. Training Data CSV
# train_csv_file contains the URL that contains the CSV file we needed

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'

# Read CSV file from the URL and print out the first five samples

train_data_name = pd.read_csv(train_csv_file) # convert CSV file to dataframe with read_csv()
train_data_name.head()

# The first column of the dataframe corresponds to the sample number. The second column is the denomination.
# The third column is the file name, and the final column is the class variable.
# The correspondence between the class variable and each denomination is as follows.
# Five Euros have y equal 0; ten Euros have y equals 1; twenty Euros have y equals 2 and so on.


# 53
print("The file name: ", train_data_name.iloc[52, 2])
print("The class label: ", train_data_name.iloc[52, 3])
# 22
print("The file name: ", train_data_name.iloc[22, 2])
print("The class label: ", train_data_name.iloc[22, 3])
# 10
print("The file name: ", train_data_name.iloc[10, 2])
print("The class label: ", train_data_name.iloc[10, 3])



# 2) Load Image

# Save the image folderpath in a variable

train_data_dir = '/resources/data/training_data_pytorch/'

# Plot the 53rd training image

train_image_name = train_data_dir + train_data_name.iloc[52, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

# Plot the 23rd image

train_image_name = train_data_dir + train_data_name.iloc[22, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

# Plot the 10th

train_image_name = train_data_dir + train_data_name.iloc[10, 2]
image = Image.open(train_image_name)
plt.imshow(image)
plt.show()

# answer = 200, 20, 10

# Question 2.2

# print out samples =[22, 32, 45] from training data

# do same code but change values

# answer = 20, 50, 100

