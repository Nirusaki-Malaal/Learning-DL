from torchvision.transforms import v2
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.optim as optimizers
from torch.nn import CrossEntropyLoss
from alexnet import AlexNet
import torch, os, signal

def handle_signal(signum,frame):
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

signal.signal(signal.SIGINT, handle_signal)

train_transform = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomCrop(32, padding=4),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.Resize((227,227)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ## these are the means and std of image net
    ]
)

test_transform = v2.Compose(
    [
        v2.Resize((227, 227)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)
# Override the default URL with a fast mirror to avoid slow downloads/timeouts from cs.toronto.edu
datasets.CIFAR10.url = "https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz"

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)


train_dataloader = DataLoader(train_dataset , batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64 , shuffle=False)
epochs = 30

device = "cuda"
alpha = 1e-3
model = AlexNet(alpha=alpha)
model = model.to(device)

if os.path.exists("./model.pth"):
    model.load_state_dict(torch.load("model.pth"))


optimizer = optimizers.Adam(model.parameters(), lr=alpha, weight_decay=1e-4) ## better minima approach  with weight decay the parameters will reach optimum better way
scheduler = optimizers.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
loss_fx = CrossEntropyLoss()

def train(dataloader, model , loss_fx , optimizer):
    size = len(dataloader.dataset)
    print(size)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Error
        pred = model(X)
        loss = loss_fx(pred , y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss , correct = 0 , 0
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fx, optimizer)
    test(test_dataloader, model, loss_fx)
    scheduler.step() ## change the learning rate here
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")