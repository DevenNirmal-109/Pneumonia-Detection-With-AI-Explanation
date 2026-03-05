import torch
import torch.nn as nn

class PneumoniaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )

        self.classifier = nn.Linear(128, 1)
        self.bbox_head = nn.Linear(128, 4)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        obj_logit = self.classifier(x)
        bbox = torch.sigmoid(self.bbox_head(x))

        return obj_logit, bbox


