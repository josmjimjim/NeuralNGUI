import os, argparse
import io
import torch
import pandas as pd
import seaborn as sns
from torch import nn
from collections import Counter, OrderedDict
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class NeuralNetwork(object):

    # Dictionary of classifier parameters
    __dict_cp = {
        'resnet18': (512, 128),
        'alexnet': (9216, 4096),
        'vgg16': (25088, 4096),
        'resnet34': (512, 128),
        'inceptionv3': (2048, 512),
        'googlenet': (1024, 256),
        'vgg19': (25088, 4096),
        'mobilenetv2': (1280, 320),
        'mobilenetv3large': (960, 240),
        'mobilenetv3small': (576, 144),
        'resnet152': (2048, 512),
        'wideresnet50': (2048, 512),
        'mnasnet': (1280, 320),
    }

    # Tuple with classifier definitions
    __dict_fc = ('resnet18', 'inceptionv3',
                'googlenet','resnet34',
                'resnet152', 'wideresnet50')

    __dict_clsf = ('alexnet', 'vgg16',
                   'mobilenetv2', 'mobilenet_v3_large',
                   'mobilenetv3small', 'mnasnet', 'vgg19')

    def __init__(self, model_name, optimizer_name,
                 batch_size, epochs, lr=0.001, save_path=None,
                 train_path=None, test_path=None,
                 weights_path=None,):

        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.train_path = train_path
        self.test_path = test_path
        self.weights_path = weights_path
        self.model = None

    def transform_images(self, image_size, mean=(0, 0, 0), std=(1, 1, 1)):
        # Define transformations: 1 to tensor and 2 normalize
        transform_list = [transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        # Define transform function with a list of parameters
        transform = transforms.Compose(transform_list)

        return transform

    def load_dataset(self, tform):
        # Load Dataset with ImageFolder
        try:
            # ImageFolder automatically converts images to RGB
            train_data = datasets.ImageFolder(self.train_path, transform=tform)
            # Number of train data
            count, data_len = self.getnumber_imagesdataset(train_data)
            num_class = self.getnumber_classes(train_data)

            if self.test_path is not None:
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
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)


            # Dict of loader
            __dict_loader = {
                'TrainLoader': train_loader,
                'TestLoader': test_loader,
                }

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
            total_mean, total_std = tuple(total_mean.tolist()), tuple(total_std.tolist())
            return total_mean, total_std

        except Exception:
            return None, None

    def load_normalized_datset(self, img_size=224):
        tform = self.transform_images(img_size)
        img, _, _, _ = self.load_dataset(tform)
        mean, std = self.compute_mean_std(img)
        tform = self.transform_images(img_size, mean=mean, std=std)
        image_loader, count, data_len, num_class = self.load_dataset(tform)
        # Dictionary with parameters
        __dict_load = OrderedDict([
            ('Number of images in dataset', data_len),
            ('Number of images per class', count),
            ('Number of classes', num_class),
            ('Mean', mean), ('Std', std),
        ])

        print('Dataset load and nomalized: Completed\n',
              'Dataset statistics:\n',
              '===================\n',
              '\tNumber of images {}\n'.format(data_len),
              '\tNumber of classes: {}\n'.format(num_class),
              '\tNumber of classe-images:\n{}\n'.format(count),
              '\tDataset mean: {}\n'.format(mean),
              '\tDataset std: {}\n'.format(std)
              )
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

        name = self.model_name

        # Check model selected
        if name == r'resnet18':
            self.model = models.resnet18()
        elif name == r'alexnet':
            self.model = models.alexnet()
        elif name == r'vgg16':
            self.model = models.vgg16()
        elif name == r'resnet34':
            self.model = models.resnet34()
        elif name == r'inceptionv3':
            self.model = models.inception_v3()
        elif name == r'googlenet':
            self.model = models.googlenet()
        elif name == r'vgg19':
            self.model = models.vgg19()
        elif name == r'mobilenetv2':
            self.model = models.mobilenet_v2()
        elif name == r'mobilenetv3large':
            self.model = models.mobilenet_v3_large()
        elif name == r'mobilenetv3small':
            self.model = models.mobilenet_v3_small()
        elif name == r'resnet152':
            self.model = models.resnet152()
        elif name == r'wideresnet50':
            self.model = models.wide_resnet50_2()
        elif name == r'mnasnet':
            self.model = models.mnasnet1_0()
        else:
            self.model = None

        model_dict = self.model.state_dict()

        if self.weights_path:
            if name not in('inceptionv3', 'googlenet'):
                with open(self.weights_path, 'rb') as f:
                    buffer = io.BytesIO(f.read())
                    buffer.seek(0)
                    f.close()
                # Filter prameters
                pretrained_dict = torch.load(buffer)
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                buffer.close()
                # Overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # Load the new state dict
                self.model.load_state_dict(model_dict)

                for param in self.model.parameters():
                    param.requires_grad = False
                print('Model: {}\nModel status: Pretrained'.format(name))
        else:
            print('Model: {}\nModel status: Not Pretrained'.format(name))

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
                if name == 'inceptionv3':
                    self.model.AuxLogits.fc = nn.Sequential(
                        OrderedDict([('fc1', nn.Linear(768, num_class)),
                                     ('output', nn.LogSoftmax(dim=1)),
                                     ]))
            elif name in self.__dict_clsf:
                self.model.classifier = classifier
            else:
                self.model = None

            print('Set up classifier')

        except Exception:
            self.model = None

    def select_optimizer(self):
        optim = self.model
        opt_name, lr = self.optimizer_name, self.lr

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

        print('Optimizer: {}'.format(opt_name))
        return optimizer

    def validate_model(self, valid_data, criterion, device):

        val_loss, accuracy = 0, 0
        y_pred, y_true = [], []
        # Iterate around all images in validation dataset
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)

            output = self.model.forward(img)
            y_pred.extend(torch.max(output).numpy())  # Save Prediction
            y_true.extend(label)  # Save Truth
            val_loss += criterion(output, label).item()

            probabilities = torch.exp(output)

            equality = (label.data == probabilities.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return val_loss, accuracy, y_pred, y_true

    def train_model(self, device='cpu'):
        # Save losses for plotting them
        running_loss_list, val_loss_list = [], []
        epochs_list = []

        # Define/calculate parameters
        steps, running_loss = 0, 0
        dataset, params = self.load_normalized_datset(img_size=224)
        epochs = self.epochs

        # Define parameters
        num_class = params['Number of classes']
        train_dataset = dataset['TrainLoader']
        valid_dataset = dataset['TestLoader']

        self.select_model()
        self.model_classifier(num_class)
        optimizer = self.select_optimizer()
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
                running_loss_list.append(np.mean(running_loss))

            # Validate model with not grad for speed
            self.model.eval()

            with torch.no_grad():
                val_loss, accuracy, y_pred, y_true = self.validate_model(valid_dataset, criterion, device)
                val_loss_list.append(val_loss)

            print('Epoch: {}/{} '.format(j + 1, epochs),
                  '\tTraining Loss: {:.3f} '.format(running_loss / steps),
                  '\tValidation Loss: {:.3f} '.format(val_loss / len(valid_dataset)),
                  '\tValidation Accuracy: {:.3f} '.format(accuracy / len(valid_dataset)),
                  )

            running_loss, steps = 0, 0

            epochs_list.append(j)

        print('Test Accuracy of the model: {:.6f} %'
              .format(100 * accuracy/len(valid_dataset)))

        # Save
        model_path = os.path.normpath(os.path.join(self.save_path,
                                                  self.model_name + '.pth'))
        torch.save(self.model.state_dict(), model_path)

        # Plot results
        plt.plot(running_loss_list, label='Training loss')
        plt.plot(val_loss_list, label='Validation loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        save_path = os.path.normpath(os.path.join(self.save_path, 'loss.png'))
        plt.savefig(save_path, dpi=300)

        # Confusion matrix
        cf_matrix = confusion_matrix(y_true, y_pred)
        df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 100)

        plt.figure(figsize=(12, 7))
        sns.heatmap(df_cm, annot=True)
        save_path = os.path.normpath(os.path.join(self.save_path, 'confusion.png'))
        plt.savefig(save_path, dpi=300)

if __name__ == '__main__':

    # Define args to pass to the model
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Select the model for training')
    parser.add_argument('optimizer', help='Select the optimizer')
    parser.add_argument('batch', help='Define the batch size')
    parser.add_argument('epochs', help='Define the epochs for training model')
    parser.add_argument('lr', help='Define learning rate for the optimizer')
    parser.add_argument('save', help='Select file directory to save model')
    parser.add_argument('train', help='Select file directory of training images')
    parser.add_argument('-t', '--test', help='Select file directory of test images')
    parser.add_argument('-w', '--weights', help='(Optional) Select pretrained weights')

    args = parser.parse_args()

    if args.weights:
        weights = args.weights
    else:
        weights = None

    if args.test:
        test = args.test
    else:
        test = None

    neural = NeuralNetwork(args.model, args.optimizer, int(args.batch),
                      int(args.epochs), float(args.lr), args.save,
                      args.train, test, weights
                     )
    neural.train_model()



