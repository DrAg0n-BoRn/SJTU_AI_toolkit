import torch
from torch import nn
from typing import Literal


# Custom CNN: Lighter alternative, with Native Grad-CAM Support
# inspired by shallow networks like LeNet and CIFAR-10/100)
class VisualCNN(nn.Module):
    def __init__(self, classes: Literal["2-class", "3-class"]):
        super(VisualCNN, self).__init__()
        
        # Feature Extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4 (Output: 256 channels)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        out_features = 3 if classes == "3-class" else 1
        self.classifier = nn.Linear(256, out_features)
        
        # Placeholder for gradients
        self.grads = None

    # Hook for gradients
    def activations_hook(self, grad):
        self.grads = grad

    def forward(self, x):
        # Pass through feature extractor
        x = self.features(x)
        
        # Register hook at the last convolutional layer
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        # Global Average Pooling and Flatten
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        # Classifier
        x = self.classifier(x)
        return x

    # Method for the gradient extraction
    def get_grads(self):
        return self.grads

    # Method for the activation extraction
    def get_activations(self, x):
        return self.features(x)
