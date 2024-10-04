import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

trainset =  datasets.ImageFolder(root='pytorchResized/train', transform=transform)

testset =   datasets.ImageFolder(root='pytorchResized/test', transform=transform)

batch_size = 4
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,256, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2)) 

        self.flat = nn.Flatten()

        self.fc3 = nn.Linear(256 * 64 * 64, 512)  
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(512, 2)



    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.pool2(x)

        x = self.pool3(x)

        x = self.flat(x)

        x = self.fc3(x)
        x = self.act3(x)
        x = self.drop3(x)

        x = self.fc4(x)
        return x
model = FakeModel()

print(summary(model, input_size=(3, 256, 256)))


loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

n_epochs = 20
for epoch in range(n_epochs):
    for inputs, labels in trainloader:
        # forward, backward, and then weight update
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc = 0
    count = 0
    for inputs, labels in testloader:
        y_pred = model(inputs)
        acc += (torch.argmax(y_pred, 1) == labels).float().sum()
        
        count += len(labels)
    
    acc /= count
    print(acc)
    print(count)
    
    print("Epoch %d: model accuracy %.2f%%" % (epoch, acc*100))



torch.save(model.state_dict(), "fakerealmodel.pth")