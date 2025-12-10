import torch
import torch.nn as nn
import torchvision.models as models

class ResNet34(nn.Module):
    def __init__(self, num_classes=1, weights='IMAGENET1K_V1'):
    # def __init__(self, num_classes=1):
        super(ResNet34, self).__init__()
        self.model = models.resnet34(weights=weights)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 1)
        )

    def forward(self, x):
        return self.model(x)

# To test the model definition:
if __name__ == "__main__":
    image = torch.randn(4, 3, 64, 64)

    model = ResNet34()

    # input image to model
    output = model(image)