import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sn
import time
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils

from classifier import *
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
BATCH_SIZE = 8
EPOCHS = 1500
LEARNING_RATE = 1e-4

DATA_SETS = ['smartwatches', 'toothbrushes']
MODEL_TYPES = ['alexnet', 'custom', 'googlenet', 'resnet18', 'vgg16']


class CustomTransforms:
    class RandomGaussianNoise:
        def __init__(self, magnitude=0.3, clamp=1.0):
            self.magnitude = magnitude
            self.clamp = clamp

        def __call__(self, tensor):
            result = tensor + torch.randn(tensor.size()) * self.magnitude
            return torch.clamp(result, max=self.clamp)

        def __repr__(self):
            return f"{self.__class__.__name__}(magnitude={self.magnitude}, clamp={self.clamp})"


def get_model(model_type, transfer_learning=False, feature_extraction=False):
    model = torch.hub.load('pytorch/vision:v0.10.0', model_type, pretrained=transfer_learning)

    if feature_extraction:
        for _, parameter in model.named_parameters():
            parameter.requires_grad = False

        if model_type in ['alexnet', 'vgg16']:
            for _, parameter in model.classifier[-1].named_parameters():
                parameter.requires_grad = True
        elif model_type in ['googlenet', 'resnet18']:
            for _, parameter in model.fc:
                parameter.requires_grad = True

    return model


def train_model(model, data_set, output_directory):
    n_classes = len(LABELS[data_set])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    transformations = transforms.Compose([
        transforms.RandomAffine(degrees=180),
        transforms.RandomApply([transforms.GaussianBlur(3)], 0.5),
        transforms.RandomApply([transforms.ColorJitter(0.5, 0.3, 0.1)], 0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        CustomTransforms.RandomGaussianNoise()
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join('data', data_set, 'train'),
        transform=transformations
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join('data', data_set, 'test'),
        transform=transformations
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    best_accuracy = 0.0
    start_time = time.time()

    time_stamps = []
    train_accuracies = []
    train_losses = []
    test_accuracies = []

    image_examples_plotted = False

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            if not image_examples_plotted:
                images = inputs.cpu()[:16]
                grid = torchvision.utils.make_grid(images, nrow=4)
                plt.figure()
                plt.imshow(np.transpose(grid, (1, 2, 0)))
                plt.savefig(os.path.join(output_directory, 'examples.png'))
                plt.tight_layout()
                plt.close()

                image_examples_plotted = True

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)
        train_loss = running_loss / len(train_data_loader)
        train_losses.append(train_loss)

        correct = 0
        total = 0

        model.eval()

        confusion_matrix = torch.zeros(n_classes, n_classes)

        for i, (inputs, labels) in enumerate(test_data_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        test_accuracy = correct / total
        test_accuracies.append(test_accuracy)

        scheduler.step()

        time_stamp = time.time() - start_time
        time_stamps.append(time_stamp)

        print(f"\nEpoch {epoch + 1}/{EPOCHS} ({time_stamp:.3f}s since start): "
              f"\n\tTraining:"
              f"\n\t\tLoss: {train_loss}"
              f"\n\t\tAccuracy: {(train_accuracy * 100):.3f}%"
              f"\n\tValidation:"
              f"\n\t\tAccuracy: {(test_accuracy * 100):.3f}%"
              f"\n\t\tBest so far: {(best_accuracy * 100):.3f}%")

        if test_accuracy >= best_accuracy:
            print(f"\tBest test accuracy yet! Saving model...")
            torch.save(model.state_dict(), os.path.join(output_directory, 'model.pt'))
            confusion_matrix = pd.DataFrame(confusion_matrix.tolist(), index=LABELS[data_set], columns=LABELS[data_set])
            confusion_matrix.to_csv(os.path.join(output_directory, 'confusion_matrix.csv'))

            plt.figure()
            sn.heatmap(confusion_matrix, annot=True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_directory, 'confusion_matrix.png'), bbox_inches='tight')
            plt.close()

            best_accuracy = test_accuracy

    data = {
        'Epoch': range(1, EPOCHS + 1),
        'Time Stamp': time_stamps,
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies
    }

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_directory, 'training_data.csv'), index=False)

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_losses, 'b', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(-50, EPOCHS + 50)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'loss.png'))
    plt.close()

    plt.figure()
    plt.plot(range(1, EPOCHS + 1), train_accuracies, 'r', label='Training Accuracy')
    plt.plot(range(1, EPOCHS + 1), test_accuracies, 'g', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(-50, EPOCHS + 50)
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'accuracies.png'))
    plt.close()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-ds', '--data_set', type=str, required=False, default='smartwatches', choices=DATA_SETS,
                            help="Which data set to use.")
    arg_parser.add_argument('-fe', '--feature_extraction', dest='feature_extraction', action='store_true',
                            help="Whether to freeze all except the classification layer. "
                                 "Only useful in Transfer Learning. ")
    arg_parser.add_argument('-o', '--output', type=str, required=False, default=os.path.join('models', 'new'),
                            help="Path of the desired output folder.")
    arg_parser.add_argument('-sd', '--state_dict', type=str, required=False, default=None,
                            help="Path to the state dictionary the model should initially load from. "
                                 "For use in transfer learning custom models.")
    arg_parser.add_argument('-tl', '--transfer_learning', dest='transfer_learning', action='store_true',
                            help="Whether transfer learning should be utilized.")
    arg_parser.add_argument('-t', '--type', type=str, required=False, default='custom',
                            choices=MODEL_TYPES, help="The type of classifier to be used.")
    arg_parser.set_defaults(feature_extraction=False, transfer_learning=False)
    args = arg_parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.type == 'custom':
        if args.transfer_learning:
            model = Classifier(16)

            if args.state_dict is None:
                print(f"When using a custom model with transfer learning, "
                      f"a state dictionary location needs to be specified!")
                return

            model.load_state_dict(torch.load(args.state_dict))
            model.final[-1] = nn.Linear(in_features=(4 * 4 * 736), out_features=19)

            args.data_set = 'toothbrushes'
        else:
            if args.data_set == 'smartwatches':
                model = Classifier(16)
            else:
                model = Classifier(19)

        if args.feature_extraction:
            for _, param in model.named_parameters():
                param.requires_grad = False

            for _, param in model.final[-1].named_parameters():
                param.requires_grad = True
    else:
        model = get_model(args.type, transfer_learning=args.transfer_learning,
                          feature_extraction=args.feature_extraction)

    model = model.to(device)
    train_model(model=model, data_set=args.data_set, output_directory=args.output)


if __name__ == '__main__':
    main()
