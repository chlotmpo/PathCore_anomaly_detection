import torch
import torch.nn as nn

"""
    RESNET50

    This network is implemented as a version of a subset of the nn.Module class from Pytorch library
    This network is composed of convolutional layers, batch normalization layers, the ReLU activation function and fully connected layer. 
    The ResNet50 architecture is composed of several residual blocks, each with a different number of filters.
    Each residual block contain 2 convolutional layer with batch normalization and ReLU activation between them. 
    The input is passed through these layers and then added back to the original input before moving on to the next block. 
"""
class Bottleneck(nn.Module):
        
    # Bottleneck is a class that defines one bottleneck layer of the ResNet architecture. A bottleneck layer is made up of three convolutional layers: 
    # a 1x1 convolutional layer that reduces the number of channels, a 3x3 convolutional layer that maintains the spatial dimensions, 
    # and a final 1x1 convolutional layer that increases the number of channels again. 
    # The output of the 3x3 convolutional layer is passed through a batch normalization layer and a ReLU activation function before being fed into the final 1x1 convolutional layer.

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        # First batch normalization layer
        self.bn1 = nn.BatchNorm2d(planes)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                            padding=1, bias=False)
        
        # Second batch normalization layer
        self.bn2 = nn.BatchNorm2d(planes)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)

        # Third batch normalization layer
        self.bn3 = nn.BatchNorm2d(planes * 4)

        # ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        # Apply the first convolutional and the first batch normalization layers
        out = self.conv1(x)
        out = self.bn1(out)

        # Apply the ReLU activation 
        out = self.relu(out)

        # Apply the second convolutional and the second btach normalization layers
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply the ReLU activation
        out = self.relu(out)

        # Apply the third convolutional and batch normalization layers
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        # Apply the ReLU activation
        out = self.relu(out)

        return out

class ResNet50(nn.Module):

  # Architecture of the ResNet-50 neural network model
  def __init__(self, output_indices = None, num_classes=1000):
    super(ResNet50, self).__init__()
    self.output_indices = output_indices

    # First convolutional layer
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # First batch normalization layer
    self.bn1 = nn.BatchNorm2d(64)

    # ReLU activation function
    self.relu = nn.ReLU(inplace=True)

    # Maxpooling layer to reduce the spatial dimensions of the input while retaining the most important information
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # Create the 4 block layers of the ResNet50 architecture using the Bottleneck class
    self.layer1 = nn.Sequential(
        Bottleneck(64, 64, downsample=nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),nn.BatchNorm2d(256))),
        Bottleneck(256, 64),
        Bottleneck(256, 64))
    
    self.layer2 = nn.Sequential(
        Bottleneck(256, 128, stride=2, downsample=nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(512))),
        Bottleneck(512, 128),
        Bottleneck(512, 128),
        Bottleneck(512, 128))

    self.layer3 = nn.Sequential(
        Bottleneck(512, 256, stride=2, downsample=nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(1024))),
        Bottleneck(1024, 256),
        Bottleneck(1024, 256),
        Bottleneck(1024, 256),
        Bottleneck(1024, 256))

    self.layer4 = nn.Sequential(
        Bottleneck(1024, 512, stride=2, downsample=nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(2048))),
        Bottleneck(2048, 512),
        Bottleneck(2048, 512),
        Bottleneck(2048, 512))

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # Fully-connected layer
    self.fc = nn.Linear(2048, num_classes)

  def forward(self, x):

    # Apply the first convolutional and the first batch normalization layers
    x = self.conv1(x)
    x = self.bn1(x)

    # Apply the ReLU activation
    x = self.relu(x)

    # Apply the maxpooling layer
    x = self.maxpool(x)

    # Apply the 4 block layers of the ResNet50 architecture
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    # Average pooling
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)

    # Apply the fully-connected layer to produce the output
    x = self.fc(x)

    return x

def extract_features(self, x):
    # Run the input tensor through the forward pass of the ResNet50 model
    output = self.forward(x)
    # If output_indices is provided, return the specified layers
    if self.output_indices is not None:
        output = [output[i] for i in self.output_indices]
    return output

# resnet50_model = ResNet50(num_classes=1000)


"""
  WIDE RESNET 50 
  This is a convolutional neural network architecture developped in 2016. It is a variant from the ResNet architecture, developed in 2015. 
  The WideResNet50 architecure is characterized by its wide residual blocks, which have a large number of filters compared to the original ResNet architecture. 
  This model can learn more complex feature from the input. It has a deeper network structure, with 50 convolutional layers. 
  It has been successful in a number of image classification tasks and has been widely used in research and industry. 
"""

"""
  Define a WideResNetBlock 
  This class is used to define a block of layers in the WideResNet model. In each block, there are 2 convolutional layers, 2 batch normalization layers, the ReLU activation function,
  a shortcut connection and a dropout layer to help prevent overfitting

  The shortcut connection is used to improve the performance of the model by allowing it to learn more complex relationship between input and output. They make easier the 
  backpropagation of the gradients, which can improve the efficiency. As a definition, this is a skip connection that allow the input to bypass one or more layers and can be directly 
  added to the output of the block. In the WideResNetBlock, it corresponds to a convolution and a batch normalization layer. 
"""
class WideResNetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride, dropout=0.2):
    super(WideResNetBlock, self).__init__()

    # First convolution layer
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

    # First batch normalization layer
    self.bn1 = nn.BatchNorm2d(out_channels)

    # ReLU activation fonction
    self.relu = nn.ReLU(inplace=True)

    # Second convolution layer
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    # Second batch normalization layer
    self.bn2 = nn.BatchNorm2d(out_channels)

    # Dropout layer
    self.dropout = nn.Dropout(dropout)

    # We have to verify the equality between the number of input and output channels, if it not the same, we add a shorcut connection
    if in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels)
      )
    else:
      self.shortcut = None

  def forward(self, x):
    # Apply the first convolutional and the first batch normalization layer 
    out = self.conv1(x)
    out = self.bn1(out)

    # Apply the ReLU activation 
    out = self.relu(out)

    # Apply the second convolutional and the second batch normalization layer
    out = self.conv2(out)
    out = self.bn2(out)

    # If a shortcut connection have been defined, we must add it to the output
    if self.shortcut is not None:
      shortcut = self.shortcut(x)
    else:
      shortcut = x

    # Add the shortcut and the output of the block and apply the ReLU activation
    out += shortcut
    out = self.relu(out)
    out = self.dropout(out)
    return out


"""
  Define the WideResNet model
  This class is used to define the overall WideResNEt model. There are one convolutional, one batch normalization layer, and 3 blocks from WideResNetNlock class.
  Then there are a global average pooling layer, and a fully-connected layer to produce the output
"""
class WideResNet(nn.Module):
  def __init__(self, depth, width_multiplier, num_classes, output_indices, dropout=0.2):
    super(WideResNet, self).__init__()

    self.output_indices = output_indices

    # Define the number of filters for each layer
    n_filters = [16, 16*width_multiplier, 32*width_multiplier, 64*width_multiplier]

    # First convolutional layer
    self.conv1 = nn.Conv2d(3, n_filters[0], kernel_size=3, stride=1, padding=1, bias=False)

    # First batch normalization layer
    self.bn1 = nn.BatchNorm2d(n_filters[0])

    # ReLU activation
    self.relu = nn.ReLU(inplace=True)

    # Create the 3 layers of WideResNet blocks
    self.layer1 = self._make_layer(WideResNetBlock, n_filters[0], n_filters[1], block_depth=depth, stride=1, dropout=dropout)
    self.layer2 = self._make_layer(WideResNetBlock, n_filters[1], n_filters[2], block_depth=depth, stride=2, dropout=dropout)
    self.layer3 = self._make_layer(WideResNetBlock, n_filters[2], n_filters[3], block_depth=depth, stride=2, dropout=dropout)

    # Define the global average pooling layer 
    self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    # Fully-connected layer
    self.fc = nn.Linear(n_filters[3], num_classes)


  def _make_layer(self, block, in_channels, out_channels, block_depth, stride, dropout):
    # First, create a list of WideResNet blocks
    layers = []

    # Add the first block to the list
    layers.append(block(in_channels, out_channels, stride, dropout))

    # Then add the other blocks
    for i in range(1, block_depth):
      layers.append(block(out_channels, out_channels, 1, dropout))

    # Return a sequential container of the blocks
    return nn.Sequential(*layers)


  def forward(self, x):

    # Apply the first convolutional layer and the first batch normalization layer
    x = self.conv1(x)
    x = self.bn1(x)

    # Apply the ReLU activation
    x = self.relu(x)

    # Apply the 3 layers of WideResNet blocks
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)

    # Apply the global average pooling 
    x = self.avg_pool(x)

    # Reshape the tensor for the fully-connected layer
    x = x.view(x.size(0), -1)

    # Apply the dully connected layer
    x = self.fc(x)

    # Extract the features
    extracted_features = [x[:, i] for i in self.output_indices]

    return x, extracted_features

# wideresnet_model = WideResNet(depth=6, width_multiplier=4, num_classes=10)