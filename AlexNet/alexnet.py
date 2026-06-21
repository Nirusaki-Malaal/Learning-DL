import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, alpha=1e-4):
        super().__init__()## must to register layers in the mode
        self.flatten = nn.Flatten()
        self.alpha = alpha
        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), # Conv1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # max pool 1
            nn.Conv2d(64,192, kernel_size=5, padding=2), # conv 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1), # conv 3
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256, kernel_size=3, padding=1), # conv 4
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #conv 5
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), # max pooling

        )
        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
        # classifier

    def forward(self, x):
        x  = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = AlexNet() 
    device  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = model.to(device)
    alpha = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)