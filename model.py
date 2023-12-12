""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
from variables import channels

def createDeepLabv3(input_channels = channels, outputchannels=1):
    """DeepLabv3 class with custom head

    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.

    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    input_channels = channels

    model = models.segmentation.deeplabv3_resnet50(pretrained=False,
                                                   progress=True)
    
    if input_channels != 3:        
        model.backbone.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias = False)

    model.classifier = DeepLabHead(2048, outputchannels)
    
    # Set the model in training mode
    model.train()
    return model