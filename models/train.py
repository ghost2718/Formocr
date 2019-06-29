from torchvision import datasets,transforms
from torch import nn,optim
import torch.nn.functional as F
from model import Classifier #importing our Classifier model defined in model.py
import torch
import matplotlib.pyplot as plt
import argparse


#ArgParser to take commandline argument for the Number of epochs to train
parser = argparse.ArgumentParser()
parser.add_argument("--num_epoch",help = "Give the number of epochs to be trained",type = int)
args = parser.parse_args()



torch.manual_seed(0)# set random seed for reproducability

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,)),
                              ])

#Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Download dataset if not present
trainset = datasets.MNIST("", download=True, train=True, transform=transform)
testset = datasets.MNIST("", download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
#initialize Classifier model
model = Classifier().to(device)
#Use SGD optimiser
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epoch = args.num_epoch

def train(epoch):
    """Training Function for every batch from the dataloader forward pass and then the backward pass of the models
    is conducted.Later the weights are updated using optimiser.step() . Metrics like loss and running loss are recoreded"""
    # loss_list = []
    runningloss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        runningloss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))
            # loss_list.append(loss.item())
    return runningloss



def test():
    """We transfer the model to Eval mode and Evalute the model on test dataset,later get metrics like trainloss and accuracy on trainset"""
    test_loss_list = []
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            ep_loss = F.nll_loss(output,target,size_average=False).item()
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss_list.append(test_loss)

        test_loss /= len(testloader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(testloader.dataset), 100. * correct / len(testloader.dataset)))
    return test_loss,100.*(correct/len(testloader.dataset))

#

loss_list = []
test_list = []
accuracy_list = []
#Training loop
for epoch in range(1, num_epoch + 1):
    loss_list.append(train(epoch))
    test_list.append(test()[0])
    accuracy_list.append(test()[1])
epoch_list = []
for i in range(1,num_epoch+1):
    epoch_list.append(i)

print(loss_list)
print(test_list)
print(accuracy_list)

torch.save(model.state_dict(),"model.pth")#Save the Model
#PLot Graphs
plt.subplot(3, 1, 1)
plt.plot(epoch_list, loss_list, color="red")
plt.title('Train Loss vs Epoch')
plt.ylabel('Train Loss')
plt.xlabel('Epoch')

plt.subplot(3, 1, 2)
plt.plot(epoch_list,test_list , color="blue")
plt.title('Test Loss vs Epoch')
plt.ylabel('Test Loss')
plt.xlabel('Epoch')

plt.subplot(3, 1, 3)
plt.plot(epoch_list, accuracy_list, color="orange")
plt.title('Accuracy vs Epoch')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.subplots_adjust(wspace = 1.2,hspace = 1.2)
plt.savefig("graph.png")
plt.show()
