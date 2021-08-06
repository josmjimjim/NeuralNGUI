import os
import torch
from torch import nn
from collections import Counter, OrderedDict
from torchvision import transforms, datasets, models # transform data


class NeuralNetwork(object):

    # Dictionary of classifier parameters
    __dict_cp = OrderedDict([
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

    # Tuple with classifier definitions
    __dict_fc = ('resnet18', 'inception_v3',
                'googlenet', 'shufflenet_v2_x1_0'
                'resnext50_32x4d', 'wide_resnet50_2')
    __dict_clsf = ('alexnet', 'vgg16', 'densenet161',
                   'mobilenet_v2', 'mobilenet_v3_large',
                   'mobilenet_v3_small', 'mnasnet1_0')

    def __init__(self, model_name, main_path,
                 train_path = None, test_path = None,):

        self.train_path = train_path
        self.test_path = test_path
        self.model_name = model_name
        self.main_path = main_path
        self.model = None

    def transform_images(self, image_size, mean=(0, 0, 0), std=(1, 1, 1)):
        # Define transformations: 1 to tensor and 2 normalize
        transform_list = [transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        # Define transform function with a list of parameters
        transform = transforms.Compose(transform_list)

        return transform

    def load_dataset(self, tform, batch_size = 64):
        # Load Dataset with ImageFolder
        try:
            # ImageFolder automatically converts images to RGB
            train_data = datasets.ImageFolder(self.train_path, transform=tform)
            # Number of train data
            count, data_len = self.getnumber_imagesdataset(train_data)
            num_class = self.getnumber_classes(train_data)

            if self.test_path != None:
                test_data = datasets.ImageFolder(self.test_path, transform=tform)
            else:
                # Length of splits
                length = len(train_data)
                train_length = int(0.8 * length)
                test_length = length - train_length

                # Split data_set
                train_data, test_data = torch.utils.data.random_split(
                    train_data, [train_length, test_length],
                    generator=torch.Generator().manual_seed(42)
                )

            # Load data as iterable
            train_loader = torch.utils.data.DataLoader(train_data, batch_size= batch_size,
                                                shuffle=True, num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size= batch_size,
                                                shuffle=True, num_workers=2)


            # Dict of loader
            __dict_loader = OrderedDict([
                ('TrainLoader',train_loader),
                ('TestLoader', test_loader),
                ])

            return __dict_loader, count, data_len, num_class

        except FileNotFoundError:
            return None, None, None, None

    def compute_mean_std(self, image_loader):
        # Store pixel sum, square pixel sum, and num of batches
        psum, psum_sq, num_batches = 0, 0, 0
        # For loop through images loaded
        image_loader = image_loader['TrainLoader']
        try:
            for inputs, _ in image_loader:
                psum += torch.mean(inputs, dim = [0, 2, 3])
                psum_sq += torch.mean(inputs**2, dim = [0, 2, 3])
                num_batches += 1
            # Compute mean and std of dataset
            total_mean = psum / num_batches
            total_var = (psum_sq / num_batches) - (total_mean ** 2)
            # Float array must be convert to tensor for speed up calculations
            total_std = torch.sqrt(torch.FloatTensor(total_var))
            return total_mean, total_std
        except Exception:
            return None, None

    def load_normalized_datset(self, img_size = 224, batch_size = 64):
        tform = self.transform_images(img_size)
        img, _, _, _ = self.load_dataset(tform, batch_size=batch_size)
        mean, std = self.compute_mean_std(img)
        tform = self.transform_images(img_size, mean=mean, std=std)
        image_loader, count, data_len, num_class = self.load_dataset(tform,
                                                     batch_size=batch_size)
        # Dictionary with parameters
        __dict_load = OrderedDict([
            ('Number of images in dataset', data_len),
            ('Number of images per class', count),
            ('Number of classes', num_class),
            ('Mean', mean), ('Std', std),
        ])

        return image_loader, __dict_load

    @staticmethod
    def getnumber_imagesdataset(data):
        data_count = Counter(data.targets)
        data_len = sum(data_count)
        return data_count, data_len

    @staticmethod
    def getnumber_classes(data):
        data_count = len(data.classes)
        return data_count

    def select_model(self):
        # Get model path
        model_path = os.path.join(self.main_path, self.model_name + r'.pth')
        name = self.model_name
        # Check model selected
        if name == 'resnet18':
            self.model = models.resnet18()
        elif name == 'alexnet':
            self.model = models.alexnet()
        elif name== 'vgg16':
            self.model = models.vgg16()
        elif name == 'densenet':
            self.model = models.densenet161()
        elif name == 'inception_v3':
            self.model = models.inception_v3()
        elif name == 'googlenet':
            self.model = models.googlenet()
        elif name == 'shufflenet':
            self.model = models.shufflenet_v2_x1_0()
        elif name == 'mobilenet_v2':
            self.model = models.mobilenet_v2()
        elif name == 'mobilenet_v3_large':
            self.model = models.mobilenet_v3_large()
        elif name == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small()
        elif name == 'resnext50_32x4d':
            self.model = models.resnext50_32x4d()
        elif name == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2()
        elif name == 'mnasnet':
            self.model = models.mnasnet1_0()
        else:
            self.model = None



    def model_classifier(self, num_class):
        # Define the last layer of the classifier or model
        # Check model selected is fc or clssifier
        name = self.model_name
        try:
            num1, num2 = self.__dict_cp[name]

            classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(num1, num2, bias=True)),
                                                    ('relu1', nn.ReLU(inplace=True)),
                                                    ('drop1', nn.Dropout(p=0.5, inplace=False)),
                                                    ('fc2', nn.Linear(num2, num2, bias=True)),
                                                    ('relu2', nn.ReLU(inplace=True)),
                                                    ('drop2', nn.Dropout(p=0.5, inplace=False)),
                                                    ('fc3', nn.Linear(num2, num_class, bias=True)),
                                                    ('output', nn.LogSoftmax(dim=1)),
                                                    ]))

            if name in self.__dict_fc:
                self.model.fc = classifier
            elif name in self.__dict_clsf:
                self.model.classifier = classifier
            else:
                self.model = None
        except Exception:
            self.model = None
            pass

    def select_optimizer(self, opt_name, lr=0.001):
        optim = self.model
        # Select optimizer by name
        if opt_name == 'Adam':
            optimizer = torch.optim.Adam(optim.parameters(), lr=lr)
        elif opt_name == 'SGD':
            optimizer = torch.optim.SGD(optim.parameters(), lr=lr)
        elif opt_name == 'LBFGS':
            optimizer = torch.optim.LBFGS(optim.parameters(), lr=lr)
        elif opt_name == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(optim.parameters(), lr=lr)
        elif opt_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(optim.parameters(), lr=lr)
        elif opt_name == 'AdamW':
            optimizer = torch.optim.AdamW(optim.parameters(), lr=lr)
        else:
            optimizer = None

        return optimizer

    def validate_model(self, valid_data, criterion, device):

        val_loss, accuracy = 0, 0

        # Iterate around all images in validation dataset
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)

            output = self.model.forward(img)
            val_loss += criterion(output, label).item()

            probabilities = torch.exp(output)

            equality = (label.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return val_loss, accuracy

    def train_model(self, epochs, device='cpu'):

        if self.model == None:
            pass

        steps, running_loss = 0, 0
        dataset, params = self.load_normalized_datset(img_size=400, batch_size = 4)

        # Define parameters
        num_class = params['Number of classes']
        train_dataset = dataset['TrainLoader']
        valid_dataset = dataset['TestLoader']

        self.select_model()
        self.model_classifier(num_class)
        optimizer = self.select_optimizer('Adam', lr=0.1)
        criterion = nn.NLLLoss()

        for j in range(epochs):
            self.model.train()
            for img, label in train_dataset:
                steps += 1

                img, label = img.to(device), label.to(device)

                optimizer.zero_grad()

                output = self.model.forward(img)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Validate model with not grad for speed
            self.model.eval()

            with torch.no_grad():
                val_loss, accuracy = self.validate_model(valid_dataset, criterion, device)

            print('Epoch: {}/{} '.format(j + 1, epochs),
                  '\tTraining Loss: {:.3f} '.format(running_loss / steps),
                  '\tValidation Loss: {:.3f} '.format(val_loss / len(valid_dataset)),
                  '\tValidation Accuracy: {:.3f} '.format(accuracy / len(valid_dataset)),
                  )

            running_loss, steps = 0, 0

    def evaluate_model(self, valid_loader, device='cpu'):
        # test-the-model
        self.model.eval()  # it-disables-dropout
        with torch.no_grad():
            correct = 0
            total = 0
            for imgs, labels in valid_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = self.model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model: {} %'.format(100 * correct / total))


if __name__ == '__main__':

    a = NeuralNetwork('mobilenet_v2','/Users/josemjimenez/Desktop/NeuralNGUI/assets',
                  '/Users/josemjimenez/Desktop/NeuralNGUI/train'
                  )
    a.train_model(10)


