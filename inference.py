import argparse
import matplotlib.pyplot as plt
import numpy
import tkinter
import torchvision.transforms as transforms
from PIL.Image import Image

from classifier import *
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from tkinter import filedialog

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', type=str, required=False, default='best_model.pt')
    args = arg_parser.parse_args()

    model = Classifier().to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()

    target_layers = [
        model.resnet_1,
        model.resnet_2,
        model.resnet_3,
        model.inception_1,
        model.inception_2,
        model.inception_3
    ]

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    window = tkinter.Tk()
    window.withdraw()

    while True:
        file_name = filedialog.askopenfilename()

        if file_name == '':
            break

        image = Image.open(file_name)
        image = transformations(image)
        image = image.to(device)
        image = image.unsqueeze(0)

        np_image = Image.open(file_name)
        np_image = np_image.resize((224, 224))
        # noinspection PyTypeChecker
        np_image = numpy.array(np_image)
        np_image = np_image.astype(numpy.float32)
        np_image /= 255

        grayscale_cam = cam(input_tensor=image, targets=None, aug_smooth=True, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(np_image, grayscale_cam, use_rgb=True)

        plt.figure()
        plt.imshow(visualization)
        plt.show()

        output = nn.functional.softmax(model(image), dim=-1).tolist()[0]
        max_value = max(output)
        max_index = output.index(max_value)

        print(f"Image contains a {LABELS[max_index]} (with {(max_value * 100):.2f}% certainty).")


if __name__ == '__main__':
    main()
