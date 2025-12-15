import torch
from torch import nn
from torchvision import models, transforms
from torch.utils import data
import matplotlib.pyplot as plt
from PIL import Image
import numpy
import os
from visual_ccc.paths import PM
from ml_tools.keys import FinalizedFileKeys
from typing import Optional


# Settings
FIGSIZE = (8, 4)
DPI = 150
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.ppm', '.pgm', '.pbm', '.pfm']
SIZE_REQUIREMENT = 256      # Alexnet

# ------------------------------------------
# Custom AlexNet
def custom_alexnet():
    alexnet = models.alexnet(weights=None)
    alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    saved_in_features: int = alexnet.classifier[6].in_features # type: ignore
    alexnet.classifier[6] = nn.Linear(in_features=saved_in_features, out_features=1, bias=True) # in_features=4096
    return alexnet


# Load image PIL
def read_image_pil(path):
    # Validation
    if not os.path.isfile(path):
        return None, None
    # Check if file is an image (by extension)
    _, extension = os.path.splitext(path)
    if extension.lower() not in valid_extensions:
        return None, None
    
    filename: str = os.path.basename(path)
    original_img = Image.open(path)
    return original_img, filename


# Transform Input Image, return image_model and image_display
def transform_image(img):
    # Transformations
    transform_model = transforms.Compose([
        transforms.Resize(size=int(1.2*SIZE_REQUIREMENT)),
        transforms.CenterCrop(size=SIZE_REQUIREMENT),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    
    transform_display = transforms.Compose([
        transforms.Resize(size=int(1.2*SIZE_REQUIREMENT)),
        transforms.CenterCrop(size=SIZE_REQUIREMENT),
        transforms.ToTensor(),
    ])
    
    # Transform image for the model
    img_model = transform_model(img).unsqueeze(0)
    dataset = data.TensorDataset(img_model, torch.zeros(size=(1,1)))
    dataloader = data.DataLoader(dataset=dataset, batch_size=1)
    img_model, _ = next(iter(dataloader))
    
    # Transform image for display
    img_display = transform_display(img)
    img_display = transforms.ToPILImage()(img_display)
    
    return img_model, img_display


# Load model + model weights. Insert hook.
def create_model():
    # Load alexnet
    alexnet = custom_alexnet()
    
    # Load weights in a system-independent way
    model_state_path = PM.model_weights
    trained_weights_dict: dict = torch.load(model_state_path, map_location=torch.device('cpu'))
    alexnet.load_state_dict(trained_weights_dict[FinalizedFileKeys.MODEL_WEIGHTS])
    
    # Load class_map if present
    class_map: Optional[dict[str,int]] = None
    if FinalizedFileKeys.CLASS_MAP in trained_weights_dict.keys():
        class_map = trained_weights_dict[FinalizedFileKeys.CLASS_MAP]

    # Insert a hook into the model
    class AlexnetHook(nn.Module):
        def __init__(self):
            super().__init__()
            # structure until last conv layer
            self.cnn = alexnet.features[:12]
            # last pooling layer
            self.lastpool = nn.Sequential(alexnet.features[12], alexnet.avgpool)
            # classifier (fully connected network)
            self.ann = alexnet.classifier
            # Placeholder for gradients
            self.grads = None

        # Hook for gradients
        def activations_hook(self, grad):
            self.grads = grad

        def forward(self, x):
            x = self.cnn(x)
            # register hook
            hook = x.register_hook(self.activations_hook)
            # resume cnn
            x = self.lastpool(x)
            # resume ann
            x = x.view(1,-1)
            x = self.ann(x)
            return x

        # method for the gradient extraction
        def get_grads(self):
            return self.grads

        # method for the activation extraction
        def get_activations(self, x):
            return self.cnn(x)
    
    # Create and return an instance
    return AlexnetHook(), class_map
    

# Get gradients and activations
def get_gradients(img_model, model, class_map):
    index_to_str = {v:k for k,v in class_map.items()}
    
    # model.eval()
    logits = model(img_model)
    
    # Get the single score from the model output (shape [1, 1])
    score = logits[0, 0]
    
    # Prediction
    if score <= 0:
        prediction = index_to_str.get(0, "Class 0")
        # Backpropagate the negative score to see what makes it "more negative"
        target_for_grad = -score
    else:
        prediction = index_to_str.get(1, "Class 1")
        # Backpropagate the positive score to see what makes it "more positive"
        target_for_grad = score
    
    # Backpropagate the correct target
    target_for_grad.backward()

    # Get gradients
    gradients = model.get_grads()

    # Pool the gradients across the channels
    average_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get activations of the last convolutional layer
    activations_avg = model.get_activations(img_model).detach()

    # weight the output channels (final layer) by the corresponding gradients
    # Reshape grads to [1, 256, 1, 1] for broadcasting
    average_gradients = average_gradients.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    # Multiply activations [1, 256, H, W] by grads [1, 256, 1, 1]
    activations_avg *= average_gradients
        
    return activations_avg, prediction


def get_gradients_multiclass(img_model, model, class_map):
    index_to_str = {v:k for k,v in class_map.items()}

    # model.eval()
    # Logits will have shape [1, 3] from a 3-class model
    logits = model(img_model)
    
    # Prediction:
    # Find the index of the highest score
    class_pred_tensor = logits.argmax(dim=1)
    class_pred = class_pred_tensor.item()
    
    # Get the string name for the predicted class
    prediction = index_to_str.get(class_pred, "Unknown")
    
    # Backpropagate the score for the predicted class
    # This tells us which pixels contributed most to *this specific class*
    logits[:, class_pred].backward()

    # Get gradients
    gradients = model.get_grads()

    # Pool the gradients across the channels
    average_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # Get activations of the last convolutional layer
    activations_avg = model.get_activations(img_model).detach()

    # weight the output channels (final layer) by the corresponding gradients
    output_channels = SIZE_REQUIREMENT
    for i in range(output_channels):
        activations_avg[:, i, :, :] *= average_gradients[i]
        
    return activations_avg, prediction


# Prepare the heatmap
def process_heatmap(activations, img_display):
    ## Heatmap
    # Average the channels of the activations
    heatmap_avg = torch.mean(activations, dim=1).squeeze()

    # Apply a Relu on the heatmap
    heatmap_avg = nn.functional.relu(heatmap_avg)

    # Normalize values
    heatmap_avg /= torch.max(heatmap_avg)
    
    # Resize heatmap to match image dimensions
    heatmap_resized = Image.fromarray(heatmap_avg.cpu().numpy()).resize(img_display.size, Image.BICUBIC) # type: ignore

    # Convert heatmap to numpy array
    heatmap_resized_np = numpy.array(heatmap_resized)
    
    return heatmap_resized_np


# Plot Grad-cam
def plot_gradcam(img_display, heatmap_resized_np):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE, dpi=DPI)
    
    # Plot original image
    axs[0].imshow(img_display, cmap='gray')
    axs[0].title.set_text("Input Image")
    axs[0].axis('off')
    
    # Plot heatmap overlapping image
    axs[1].imshow(img_display, alpha=0.5, cmap='gray')
    axs[1].imshow(heatmap_resized_np, alpha=0.5, cmap='jet')
    axs[1].title.set_text("Grad-CAM Heatmap")
    axs[1].axis('off')
    
    # Adjust subplot params to reduce padding
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.90, wspace=0.1)
    
    return fig
