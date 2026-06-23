import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes=10, alpha=1e-4):
        super().__init__()## must to register layers in the mode
        self.flatten = nn.Flatten()
        self.alpha = alpha
        # feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Conv1
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # max pool 1

            nn.Conv2d(32,64, kernel_size=3, padding=1), # conv 2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), ## 7x7x64

            nn.Conv2d(64,128, kernel_size=3, padding=1), # conv 2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),            
            
        )
        self.classifier = nn.Sequential(
    nn.Linear(128 * 7 * 7, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.25),
    nn.Linear(1024, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.25),
    nn.Linear(256, 10)
)
        # classifier

    def forward(self, x):
        x  = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
if __name__ == "__main__":
    model = Model() 
    device  = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model = model.to(device)
    alpha = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)