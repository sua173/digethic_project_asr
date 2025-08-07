# Define model initialization functions

import numpy as np


def lecun_initialization(model):
    """Execute LeCun parameter initialization method
    Initialize each weight (excluding bias) with random values
    from normal distribution with mean 0 and std 1/sqrt(dim)
    (dim is the input dimension)
    model: Model defined in PyTorch
    """
    # Extract model parameters in order and execute initialization
    for param in model.parameters():
        # Extract parameter values
        data = param.data
        # Extract tensor dimension of parameter
        dim = data.dim()
        # Process based on dimension
        if dim == 1:
            # dim = 1 indicates bias component
            # Initialize to zero
            data.zero_()
        elif dim == 2:
            # dim = 2 indicates linear projection weight matrix
            # Get input dimension = size(1)
            n = data.size(1)
            # Initialize with normal distribution random values
            # using reciprocal of square root of input dimension
            # as standard deviation
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 3:
            # dim = 3 indicates 1D convolution matrix
            # Initialize with normal distribution random values
            # using reciprocal of square root of
            # (input channels * kernel size) as standard deviation
            n = data.size(1) * data.size(2)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        elif dim == 4:
            # dim = 4 indicates 2D convolution matrix
            # Initialize with normal distribution random values
            # using reciprocal of square root of
            # (input channels * kernel size (rows) * kernel size (cols))
            # as standard deviation
            n = data.size(1) * data.size(2) * data.size(3)
            std = 1.0 / np.sqrt(n)
            data.normal_(0, std)
        else:
            # Other dimensions are not supported
            print("lecun_initialization: dim > 4 is not supported.")
            exit(1)
