

import torchvision.models as models
model = models.resnet18(pretrained=True) # resnet is a deeper network

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225] # standard deviation

# use particular transform depending on the model you use
# make sure size and normalisation parameter are the same:
transforms_stuff = transforms.Compose([transforms.Resize(224))
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

train_dataset = dataset(root='./data', download = True, transform=transforms_stuff) # has to be coloured image and has to have proper input for resnet
validation_dataset = dataset(root='./data', split = 'test', download = True, transform=transforms_stuff)

# set param.requires_grad=False to freeze model or you know in advance that you’re not going to use gradients w.r.t. some parameters.
# For example if you want to finetune a pretrained CNN, it’s enough to switch the requires_grad flags in the frozen base,
# and no intermediate buffers will be saved, until the computation gets to the last layer,
# where the affine transform will use weights that require gradient, and the output of the network will also require them.

for param in model.parameters():
    param.requires_grad=False
    model.fc = nn.Linear(512,3) # 3 classes, 512 inputs from hidden layer

criterion = nn.CrossEntropyLoss() # create loss function

optimiser = torch.optim.Adam([parameters for parameters in model.parameters() if parameters.requires_grad], lr=0.001)

train_loader = torch.utils.data.Dataloader(dataset = train_dataset, batch_size = 100) # create training loader
validation_loader = torch.utils.data.Dataloader(dataset = validation_dataset_dataset, batch_size = 5000) # create validation loader

for epoch in range(n_epochs):
    for x,y in train_loader:
        optimiser.zero_grad()
        z = model(x)
        loss = criterion(z,y)
        loss.backward()
        optimiser.step()
    correct = 0

    for x_test, y_test in validation_loader:
        z = model(x)
        _,yhat=torch.max(z.data,1)
        correct+=yhat==y_test.sum().item()
    accuracy=correct/N_test
    accuracy_list.append(accuracy)
    loss_list.append(loss.data)

train()
eval()




