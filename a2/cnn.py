import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def load_data(transform=None):
    transform_list = [transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                      transforms.Normalize(0.1307, 0.3081)]
    if transform:
        transform_list.append(transform)
    transform_fn = transforms.Compose(transform_list)
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform_fn, download=True)
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform_fn, download=True)

    # Cut the training set to 1/10
    # indices = np.random.choice(len(train_dataset), len(
    #     train_dataset) // 10, replace=False)
    # train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.layers(x)


def train(model, device, train_loader, optimizer, criterion, epoch, regularization=False):
    augmentation_list = [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.Lambda(lambda x: x + 1*torch.randn_like(x)),
        transforms.Lambda(lambda x: x + 0.1*torch.randn_like(x)),
        transforms.Lambda(lambda x: x + 0.01*torch.randn_like(x))
    ]
    model.train()
    train_loss = 0.0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        if regularization:
            transform_fn = random.choice(augmentation_list)
            data = transform_fn(data)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = output.argmax(1)
        correct += predicted.eq(target).sum().item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(
        f'Train Epoch: {epoch}\tLoss: {avg_loss:.6f}\tAccuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)')

    return avg_loss, accuracy


def test(model, device, test_loader, criterion, message='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        predicted = output.argmax(1)
        correct += predicted.eq(target).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'{message}: loss: {avg_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return avg_loss, accuracy


def plot_metrics(epochs, train_losses, test_losses, train_accuracies, test_accuracies, retrained=False):
    title = ' (Data Augmented)' if retrained else ''
    path = '_retrained' if retrained else ''
    if not os.path.exists(f'plots{path}'):
        os.makedirs(f'plots{path}')

    plt.figure()
    plt.plot(epochs, test_accuracies, 'bo-', label='Test Accuracy')
    plt.title(f'Test Accuracy vs Epochs{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots{path}/test_accuracy_vs_epochs.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, train_accuracies, 'go-', label='Train Accuracy')
    plt.title(f'Train Accuracy vs Epochs{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'plots{path}/train_accuracy_vs_epochs.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, test_losses, 'ro-', label='Test Loss')
    plt.title(f'Test Loss vs Epochs{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots{path}/test_loss_vs_epochs.png')
    plt.close()

    plt.figure()
    plt.plot(epochs, train_losses, 'mo-', label='Train Loss')
    plt.title(f'Train Loss vs Epochs{title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'plots{path}/train_loss_vs_epochs.png')
    plt.close()


def plot_flip_accuracies(flip_types, flip_accuracies, retrained=False):
    title = ' (Data Augmented)' if retrained else ''
    path = '_retrained' if retrained else ''
    plt.bar(flip_types, flip_accuracies, color=['red', 'green', 'blue'])
    plt.title(f'Test Accuracy vs Type of Flip{title}')
    plt.xlabel('Flip Type')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig(f'plots{path}/accuracy_vs_flip.png')
    plt.close()


def plot_noise_accuracies(noise_vars, accuracies, retrained=False):
    title = ' (Data Augmented)' if retrained else ''
    path = '_retrained' if retrained else ''
    plt.bar(noise_vars, accuracies, color=['orange', 'lightgreen', 'purple'])
    plt.title(f'Test Accuracy vs Gaussian Noise Variance{title}')
    plt.xlabel('Gaussian Noise Variance')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.savefig(f'plots{path}/accuracy_vs_noise.png')
    plt.close()


def main(regularization=False):
    train_loader, test_loader = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG11().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_accuracy = 0.0
    num_epoch = 10 if regularization else 5
    epochs = range(1, num_epoch + 1)
    test(model, device, test_loader, criterion)
    for epoch in epochs:
        train_loss, train_accuracy = train(
            model, device, train_loader, optimizer, criterion, epoch, regularization)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        best_accuracy = max(best_accuracy, test_accuracy)

    print(f'Best Accuracy: {best_accuracy:.2f}%\n')

    plot_metrics(epochs, train_losses, test_losses,
                 train_accuracies, test_accuracies, regularization)

    _, test_horizontal_flip_loader = load_data(
        transforms.RandomHorizontalFlip(p=1))
    _, test_vertical_flip_loader = load_data(
        transforms.RandomVerticalFlip(p=1))

    _, horizontal_accuracy = test(
        model, device, test_horizontal_flip_loader, criterion, 'Horizontal Flip')
    _, vertical_accuracy = test(
        model, device, test_vertical_flip_loader, criterion, 'Vertical Flip')

    flip_types = ['Original', 'Horizontal Flip', 'Vertical Flip']
    flip_accuracies = [best_accuracy, horizontal_accuracy, vertical_accuracy]

    plot_flip_accuracies(flip_types, flip_accuracies, regularization)

    noise_vars = [0.01, 0.1, 1]
    noise_accuracies = []
    for noise_var in noise_vars:
        _, test_noise_loader = load_data(transforms.Lambda(
            lambda x: x + noise_var*torch.randn_like(x)))
        _, accuracy = test(model, device, test_noise_loader,
                           criterion, f'Gaussian Noise Variance {noise_var}')
        noise_accuracies.append(accuracy)

    plot_noise_accuracies(['0.01', '0.1', '1'],
                          noise_accuracies, regularization)


if __name__ == "__main__":
    main()
    # main(True)
