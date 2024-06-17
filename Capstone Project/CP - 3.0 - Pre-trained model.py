# Import PyTorch Modules will be used in the lab

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules will be used in the lab

import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

# 1) Create Dataset Class and Object

# Url that contains CSV files

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'

# Absolute path for finding the directory contains image datasets

train_data_dir = '/resources/data/training_data_pytorch/'
validation_data_dir = '/resources/data/validation_data_pytorch/'


# Create Dateaset Class

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]

        # Get Length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y

# Construct the composed object for transforming the image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

# Create the train dataset and validation dataset

train_dataset = Dataset(transform=composed
                        ,csv_file=train_csv_file
                        ,data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)

# 1) Question 3.1: Preparation

# Step 1: Load the pre-trained model resnet18 Set the parameter pretrained to true.

model = models.resnet18(pretrained=True)

# Step 2: The following lines of code will set the attribute requires_grad to False. As a result, the parameters will not be affected by training

for param in model.parameters():
    param.requires_grad = False

# Step 3: Replace the output layer model.fc of the neural network with a nn.Linear object, to classify 7 different bills. For the parameters in_features  remember the last hidden layer has 512 neurons.

model.fc = nn.Linear(512,7)

## final model
# Import PyTorch Modules will be used in the lab

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules will be used in the lab

import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

# 1) Create Dataset Class and Object

# Url that contains CSV files

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'

# Absolute path for finding the directory contains image datasets

train_data_dir = '/resources/data/training_data_pytorch/'
validation_data_dir = '/resources/data/validation_data_pytorch/'


# Create Dateaset Class

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]

        # Get Length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y

# Construct the composed object for transforming the image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

# Create the train dataset and validation dataset

train_dataset = Dataset(transform=composed
                        ,csv_file=train_csv_file
                        ,data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)



model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512,7)

print(model)










# Question 3.2: Train the model

# Step 1: Create a cross entropy criterion function

criterion = nn.CrossEntropyLoss()

# Step 2: Create a training loader and validation loader object, the batch size is 15 and 10 respectively

train_loader = torch.utils.data.Dataloader(dataset = train_dataset, batch_size = 15) # create training loader
validation_loader = torch.utils.data.Dataloader(dataset = validation_dataset_dataset, batch_size = 10) # create validation loader

# Step 3: Use the following optimizer to minimize the loss

optimiser = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.003)

# Step 4: Train the model for 20 epochs, save the loss in a list as will as the accuracy on the validation data for every epoch. The entire process may take 6.5 minutes.
# Print the validation accuracy for each epoch during the epoch loop. Then, plot the training loss for each epoch and validation error for each epoch.

for epoch in range(N_EPOCHS):
    loss_sublist = []
    for x,y in train_loader:
        model.train()
        optimiser.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimiser.step()
        loss_list.append(np.mean(loss_sublist))

    correct = 0

    for x_test, y_test in validation_loader:
        model.eval()
        z = model(x_test)
        _,yhat=torch.max(z.data,1)
        correct+=yhat==y_test.sum().item()
    accuracy=correct/n_test
    accuracy_list.append(accuracy)


# Step 5: Plot the training loss for each iteration

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('total loss', color=color)
ax1.tick_params(axis='y', color=color)

# Step 6: Plot the validation accuracy for each epoch

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)
ax2.plot(accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()










# Question 3.3
# Create a test dataset using validation data. And, create your own plot_random_image() function to plot 5 random images which index is in the numbers list. Run the function to plot image, print the predicted label and print a string indicate whether it has been correctly classified or mis-classified.

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules will be used in the lab

import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
# 1) Create Dataset Class and Object

# Url that contains CSV files

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'

# Absolute path for finding the directory contains image datasets

train_data_dir = '/resources/data/training_data_pytorch/'
validation_data_dir = '/resources/data/validation_data_pytorch/'


# Create Dateaset Class

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]

        # Get Length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y

# Construct the composed object for transforming the image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

# Create the test dataset using validation dataset


test_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)
test_data_name = pd.read_csv(validation_csv_file) # convert CSV file to dataframe with read_csv()
test_data_name.head()


look_up = {0: 'predicted: $5'
           , 1: 'predicted: $10'
           , 2: 'predicted: $20'
           , 3: 'predicted: $50'
           , 4: 'predicted: $100'
           , 5: 'predicted $200'
           , 6: 'predicted $500'}

random.seed(0)
numbers = random.sample(range(70), 5)


for i in range(5):
    name = "/" + str(numbers + 1) + ".jpeg"
    Input = test_data_dir + name  # Input= "/resources/data/training_data_pytorch/0.jpeg"
    img = Image.open(Input)
    plt.imshow(img)
    plt.show()
    print("The predicted label: ", test_data_name.iloc[numbers, 3])
    # need model to compare to

    if yhat == test_data_name.iloc[numbers, 3]
        print("correctly classified")

    else
        print("mis-classified")














# Question 3.4

# Repeat the steps in Question 3.1, 3.2 to predict the result using models.densenet121 model. Then, print out the last validation accuracy.
# Steps:
# Load the pre-trained model Densenet
# Replace the last classification layer with only 7 classes
# Set the configuration (parameters)
# Train the model
# Print the last validation accuracy

import torch
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import pandas
from torchvision import transforms
import torch.nn as nn
torch.manual_seed(0)

# Import Non-PyTorch Modules will be used in the lab

import time
from imageio import imread
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

# 1) Create Dataset Class and Object

# Url that contains CSV files

train_csv_file = 'https://cocl.us/DL0320EN_TRAIN_CSV'
validation_csv_file = 'https://cocl.us/DL0320EN_VALID_CSV'

# Absolute path for finding the directory contains image datasets

train_data_dir = '/resources/data/training_data_pytorch/'
validation_data_dir = '/resources/data/validation_data_pytorch/'


# Create Dateaset Class

class Dataset(Dataset):

    # Constructor
    def __init__(self, csv_file, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_name = pd.read_csv(csv_file)
        self.len = self.data_name.shape[0]

        # Get Length

    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):
        img_name = self.data_dir + self.data_name.iloc[idx, 2]
        image = Image.open(img_name)
        y = self.data_name.iloc[idx, 3]
        if self.transform:
            image = self.transform(image)
        return image, y

# Construct the composed object for transforming the image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
composed = transforms.Compose([transforms.Resize((224, 224))
                               , transforms.ToTensor()
                               , transforms.Normalize(mean, std)])

# Create the train dataset and validation dataset

train_dataset = Dataset(transform=composed
                        ,csv_file=train_csv_file
                        ,data_dir=train_data_dir)

validation_dataset = Dataset(transform=composed
                          ,csv_file=validation_csv_file
                          ,data_dir=validation_data_dir)



model_des = models.densenet121(pretrained=True)

for param in model_des.parameters():
    param.requires_grad = False

model_des.fc = nn.Linear(1024,7)

print(model_des)

criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 15) # create training loader
validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = 10) # create validation loader

optimiser = torch.optim.Adam([parameters for parameters in model_des.parameters() if parameters.requires_grad], lr=0.003)

N_EPOCHS = 10
loss_list = []
accuracy_list = []
correct = 0
n_test = len(validation_dataset)


for epoch in range(N_EPOCHS):
    loss_sublist = []
    for x,y in train_loader:
        model_des.train()
        optimiser.zero_grad()
        z = model_des(x)
        loss = criterion(z,y)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimiser.step()
        loss_list.append(np.mean(loss_sublist))

    correct = 0

    for x_test, y_test in validation_loader:
        model_des.eval()
        z = model_des(x_test)
        _,yhat=torch.max(z.data,1)
        correct+=yhat==y_test.sum().item()
    accuracy=correct/n_test
    accuracy_list.append(accuracy)


print("The last validation accuracy:", accuracy_list[9]) # last validation thus index

# Save the model

torch.save(model, "resnet18_pytorch.pt")
torch.save(model_des, "densenet121_pytorch.pt")
