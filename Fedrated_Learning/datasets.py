import torch
import torchvision

def get_dataset(dir, name):
    if name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(dir, train=True, download=True,
                                                   transform = torchvision.transforms.ToTensor())
        eval_dataset = torchvision.datasets.MNIST(dir, train=False,
                                                  transform = torchvision.transforms.ToTensor())

    elif name == 'cifar':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4919, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))
        ]
        )

        transform_eval = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4919, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010))
        ])


        train_dataset = torchvision.datasets.CIFAR10(dir, download = True,train = True, transform = transform_train)
        eval_dataset = torchvision.datasets.CIFAR10(dir, train = False, transform = transform_eval, download=True)

    return (train_dataset, eval_dataset)