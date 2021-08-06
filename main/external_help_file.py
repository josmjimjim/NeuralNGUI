from collections import OrderedDict
import torch.nn as nn

num_class = 7

__dict_opt = OrderedDict([
    ('resnet18', (512, 128)),
    ('alexnet', (9216, 4096)),
    ('vgg16', (25088, 4096)),
    ('densenet161', (2208, 552)),
    ('inception_v3', (2048, 512)),
    ('googlenet', (1024, 256)),
    ('shufflenet_v2_x1_0', (1024, 256)),
    ('mobilenet_v2', (1280, 320)),
    ('mobilenet_v3_large', (960, 240)),
    ('mobilenet_v3_small', (576, 144)),
    ('resnext50_32x4d', (2048, 512)),
    ('wide_resnet50_2', (2048, 512)),
    ('mnasnet1_0', (1280, 320)),
])

# Fc layers: resnet18, inception_v3, googlenet, shufflenet_v2_x1_0
# resnext50_32x4d, wide_resnet50_2,

num1, num2 = __dict_opt['resnet18']


# Classifier layer: alexnet, vgg16, densenet161,
# mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small, mnasnet1_0

classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num1, num2, bias= True)),
                                        ('relu1', nn.ReLU(inplace=True)),
                                        ('drop1', nn.Dropout(p=0.5, inplace=False)),
                                        ('fc2', nn.Linear(num2, num2, bias=True)),
                                        ('relu2', nn.ReLU(inplace=True)),
                                        ('drop2', nn.Dropout(p=0.5, inplace=False)),
                                        ('fc3', nn.Linear(num2, num_class, bias=True)),
                                        ('output', nn.LogSoftmax(dim=1)),
                                        ]))

#Compute mean


def compute_mean_std(self, image_loader):
    # Store pixel sum, square pixel sum, and num of batches
    psum, psum_sq, num_batches = 0, 0, 0
    # For loop through images loaded
    try:
        for inputs, _ in image_loader:
            psum += torch.mean(inputs, dim=[0, 2, 3])
            psum_sq += torch.mean(inputs ** 2, dim=[0, 2, 3])
            num_batches += 1
        # Compute mean and std of dataset
        total_mean = psum / num_batches
        total_var = (psum_sq / num_batches) - (total_mean ** 2)
        # Float array must be convert to tensor for speed up calculations
        total_std = torch.sqrt(torch.tensor(total_var))
        return total_mean, total_std
    except Exception:
        return None, None














