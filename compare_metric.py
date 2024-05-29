import matplotlib.pyplot as plt
import os
import pandas as pd

ROOT_DIR = os.path.join('models', 'smartwatches')

LABEL_TRAINING = "Training Accuracy"
LABEL_VALIDATION = "Validation Accuracy"

METRIC_TRAINING = "Train Accuracy"
METRIC_VALIDATION = "Test Accuracy"


def get_standard_plot_options(subplots):
    if subplots == 2:
        plots = [{
            'y_label': LABEL_TRAINING,
            'label_x': False
        }, {
            'y_label': LABEL_VALIDATION,
            'legend': False
        }]
    elif subplots == 4:
        plots = [[{
            'y_label': LABEL_TRAINING,
            'label_x': False
        }, {
            'legend': False,
            'label_x': False,
            'label_y': False
        }], [{
            'y_label': LABEL_VALIDATION,
            'legend': False
        }, {
            'legend': False,
            'label_y': False
        }]]
    else:
        plots = [{}]

    return plots


def get_data(path, metrics=None, color='b', label='Accuracy'):
    if metrics is None:
        metrics = [{'name': METRIC_TRAINING, 'color': color, 'label': label}]

    df = pd.read_csv(os.path.join(path, 'training_data.csv'))

    data = {}

    for metric in metrics:
        metric_data = {
            'data': df[metric['name']]
        }

        if 'color' in metric.keys():
            metric_data['color'] = metric['color']
        else:
            metric_data['color'] = color

        if 'label' in metric.keys():
            metric_data['label'] = metric['label']
        else:
            metric_data['label'] = label

        data[metric['name']] = metric_data

    return data


def print_data(data, average_length=50):
    for name, metrics in data.items():
        for metric_name, metric in metrics.items():
            d = metric['data']
            temp = d.iloc[-average_length:]

            print(f"{name} ({metric_name}):")
            print(f"\t{'Max: ':20s}{d.max() * 100:.2f}%")
            print(f"\t{'Epoch: ':20s}{d.argmax() + 1}")
            print(f"\t{f'Mean ({average_length}): ':20s}{temp.mean() * 100:.2f}%")
            print(f"\t{'STD: ':20s}{temp.std() * 100:.2f}%\n")


def draw_plot(axis, plot):
    axis.plot(
        range(1, len(plot['data']) + 1),
        plot['data'],
        plot['color'],
        label=plot['label']
    )


def annotate_subplot(axis, **kwargs):
    y_label = 'Accuracy'

    legend = True
    label_x = True
    label_y = True

    x_min = -50
    x_max = 1550

    y_min = -0.1
    y_max = 1.1

    if 'y_label' in kwargs.keys():
        y_label = kwargs['y_label']

    if 'x_min' in kwargs.keys():
        x_min = kwargs['x_min']

    if 'x_max' in kwargs.keys():
        x_max = kwargs['x_max']

    if 'y_min' in kwargs.keys():
        y_min = kwargs['y_min']

    if 'y_max' in kwargs.keys():
        y_max = kwargs['y_max']

    if 'legend' in kwargs.keys():
        legend = kwargs['legend']

    if 'label_x' in kwargs.keys():
        label_x = kwargs['label_x']

    if 'label_y' in kwargs.keys():
        label_y = kwargs['label_y']

    if legend:
        axis.legend()

    if label_x:
        axis.set(xlabel='Epoch')

    if label_y:
        axis.set(ylabel=y_label)

    plt.setp(axis.get_yticklabels(), visible=True)
    axis.grid(True)
    axis.set_xlim([x_min, x_max])
    axis.set_ylim([y_min, y_max])


def finalize_figure():
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_learning_rate_comparison():
    directory = os.path.join(ROOT_DIR, 'lr_comparison')

    metrics = [{
        'name': METRIC_TRAINING
    }]

    data = {
        '2e-3': get_data(os.path.join(directory, '2e-3'), metrics=metrics, color='r', label='2e-3'),
        '1e-3': get_data(os.path.join(directory, '1e-3'), metrics=metrics, color='b', label='1e-3'),
        '1e-4': get_data(os.path.join(directory, '1e-4'), metrics=metrics, color='g', label='1e-4'),
        '8e-5': get_data(os.path.join(directory, '8e-5'), metrics=metrics, color='tab:orange', label='8e-5'),
        '1e-5': get_data(os.path.join(directory, '1e-5'), metrics=metrics, color='y', label='1e-5')
    }

    print_data(data)

    plot_options = {
        'x_min': -10,
        'x_max': 310
    }

    plt.figure()
    axs = plt.axes()

    for d in data.values():
        draw_plot(axs, d[METRIC_TRAINING])

    annotate_subplot(axs, **plot_options)
    finalize_figure()


# noinspection PyTypeChecker,PyArgumentList
def plot_optimizer_comparison():
    directory = os.path.join(ROOT_DIR, 'optimizer_comparison')

    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'Adam': get_data(os.path.join(directory, 'Adam'), metrics=metrics, color='b', label='Adam'),
        'SGDM': get_data(os.path.join(directory, 'SGDM'), metrics=metrics, color='r', label='SGDM')
    }

    print_data(data)

    plot_options = get_standard_plot_options(2)

    # Top
    plot_options[0]['x_min'] = -10
    plot_options[0]['x_max'] = 310

    # Bottom
    plot_options[1]['x_min'] = -10
    plot_options[1]['x_max'] = 310
    plot_options[1]['y_max'] = 0.6

    fig, axs = plt.subplots(2)

    for d in data.values():
        draw_plot(axs[0], d[METRIC_TRAINING])
        draw_plot(axs[1], d[METRIC_VALIDATION])

    for y in range(2):
        annotate_subplot(axs[y], **plot_options[y])

    finalize_figure()


# noinspection PyTypeChecker,PyArgumentList
def plot_scheduler_comparison():
    directory = os.path.join(ROOT_DIR, 'scheduler_comparison')

    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'Cosine': get_data(os.path.join(directory, 'Cosine'), metrics=metrics, color='b', label='Cosine'),
        'Step': get_data(os.path.join(directory, 'Step'), metrics=metrics, color='r', label='Step')
    }

    print_data(data)

    plot_options = get_standard_plot_options(2)

    # Top
    plot_options[0]['x_min'] = -10
    plot_options[0]['x_max'] = 310

    # Bottom
    plot_options[1]['x_min'] = -10
    plot_options[1]['x_max'] = 310
    plot_options[1]['y_min'] = 0.1
    plot_options[1]['y_max'] = 0.6

    fig, axs = plt.subplots(2)

    for d in data.values():
        draw_plot(axs[0], d[METRIC_TRAINING])
        draw_plot(axs[1], d[METRIC_VALIDATION])

    for y in range(2):
        annotate_subplot(axs[y], **plot_options[y])

    finalize_figure()


def plot_hyper_parameter_comparisons():
    plot_learning_rate_comparison()
    plot_optimizer_comparison()
    plot_scheduler_comparison()


def plot_affine_comparison():
    directory = os.path.join(ROOT_DIR, 'augmentation_comparison', 'Affine')

    metrics = [{
        'name': METRIC_VALIDATION
    }]

    data = {
        'none': get_data(os.path.join(directory, 'None'), metrics=metrics, color='b', label='None'),
        '0.05, 0.05, [0.9, 1.1], 15': get_data(os.path.join(directory, '005-005-09_11-15'), metrics=metrics, color='g',
                                               label='5% x, 5% y, 90-110% scale, 15° shearing'),
        '0.1, 0.1, [0.9, 1.1], 15': get_data(os.path.join(directory, '01-01-09_11-15'), metrics=metrics, color='r',
                                             label='10% x, 10% y, 90-110% scale, 15° shearing')
    }

    print_data(data)

    plot_options = {
        'x_min': -10,
        'x_max': 310,
        'y_max': 0.8
    }

    plt.figure()
    axs = plt.axes()

    for d in data.values():
        draw_plot(axs, d[METRIC_VALIDATION])

    annotate_subplot(axs, **plot_options)
    finalize_figure()


# noinspection PyTypeChecker,PyArgumentList
def plot_blur_comparison():
    directory = os.path.join(ROOT_DIR, 'augmentation_comparison', 'GaussianBlur')

    metrics = [{
        'name': METRIC_VALIDATION
    }]

    data = {
        'None': get_data(os.path.join(directory, 'None'), metrics=metrics, color='b', label='None'),
        '3x3': get_data(os.path.join(directory, '3x3 Half'), metrics=metrics, color='g',
                        label='3x3 at 50% probability'),
        '5x5': get_data(os.path.join(directory, '5x5 Half'), metrics=metrics, color='r', label='5x5 at 50% probability')
    }

    print_data(data)

    plot_options = get_standard_plot_options(2)

    # Top
    plot_options[0]['y_label'] = LABEL_VALIDATION
    plot_options[0]['x_min'] = -10
    plot_options[0]['x_max'] = 310
    plot_options[0]['y_max'] = 0.8

    # Bottom
    plot_options[1]['x_min'] = 150
    plot_options[1]['x_max'] = 300
    plot_options[1]['y_min'] = 0.4
    plot_options[1]['y_max'] = 0.65

    fig, axs = plt.subplots(2)

    for d in data.values():
        draw_plot(axs[0], d[METRIC_VALIDATION])
        draw_plot(axs[1], d[METRIC_VALIDATION])

    for y in range(2):
        annotate_subplot(axs[y], **plot_options[y])

    finalize_figure()


# noinspection PyTypeChecker,PyArgumentList
def plot_color_comparison():
    directory = os.path.join(ROOT_DIR, 'augmentation_comparison', 'ColorJitter')

    metrics = [{
        'name': METRIC_VALIDATION
    }]

    data = {
        'None': get_data(os.path.join(directory, 'None'), metrics=metrics, color='b', label='None'),
        '531': get_data(os.path.join(directory, '531'), metrics=metrics, color='g',
                        label='50% B., 30% C., 10% S., 0% H.'),
        '321': get_data(os.path.join(directory, '321'), metrics=metrics, color='y',
                        label='30% B., 20% C., 10% S., 0% H.'),
        '2211': get_data(os.path.join(directory, '2211'), metrics=metrics, color='r',
                         label='20% B., 20% C., 10% S., 10% H.')
    }

    print_data(data)

    plot_options = get_standard_plot_options(2)

    # Top
    plot_options[0]['y_label'] = LABEL_VALIDATION
    plot_options[0]['x_min'] = -10
    plot_options[0]['x_max'] = 310
    plot_options[0]['y_max'] = 0.7

    # Bottom
    plot_options[1]['x_min'] = 150
    plot_options[1]['x_max'] = 300
    plot_options[1]['y_min'] = 0.4
    plot_options[1]['y_max'] = 0.65

    fig, axs = plt.subplots(2)

    for d in data.values():
        draw_plot(axs[0], d[METRIC_VALIDATION])
        draw_plot(axs[1], d[METRIC_VALIDATION])

    for y in range(2):
        annotate_subplot(axs[y], **plot_options[y])

    finalize_figure()


def plot_crop_comparison():
    directory = os.path.join(ROOT_DIR, 'augmentation_comparison', 'Crop and Resize')

    metrics = [{
        'name': METRIC_VALIDATION
    }]

    data = {
        'resize': get_data(os.path.join(directory, 'Resize'), metrics=metrics, color='b', label='Resize'),
        'center': get_data(os.path.join(directory, 'CenterCrop'), metrics=metrics, color='g',
                           label='Tight Center Crop'),
        'bordered': get_data(os.path.join(directory, 'BorderedCenterCrop'), metrics=metrics, color='r',
                             label='Bordered Center Crop')
    }

    print_data(data)

    plot_options = {
        'x_min': -10,
        'x_max': 310,
        'y_max': 0.8
    }

    plt.figure()
    axs = plt.axes()

    for d in data.values():
        draw_plot(axs, d[METRIC_VALIDATION])

    annotate_subplot(axs, **plot_options)
    finalize_figure()


def plot_augmentation_comparisons():
    plot_affine_comparison()
    plot_blur_comparison()
    plot_color_comparison()
    plot_crop_comparison()


# noinspection PyTypeChecker
def plot_architecture_comparison(training='FineTuned'):
    if training not in ['FeatureExtraction', 'FineTuned', 'FromScratch']:
        print(f"Folder {training} not found!")
        return

    directory = os.path.join(ROOT_DIR, 'architecture_comparison', training)
    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'alexnet': get_data(os.path.join(directory, 'AlexNet'), metrics=metrics, color='y', label='AlexNet'),
        'googlenet': get_data(os.path.join(directory, 'GoogLeNet'), metrics=metrics, color='g', label='GoogLeNet'),
        'resnet18': get_data(os.path.join(directory, 'ResNet18'), metrics=metrics, color='r', label='ResNet18'),
        'vgg16': get_data(os.path.join(directory, 'VGG16'), metrics=metrics, color='tab:orange', label='VGG16')
    }

    if training == 'FromScratch':
        data['custom'] = get_data(os.path.join(directory, 'custom'), metrics=metrics, label='Custom')

    print_data(data)

    plot_options = get_standard_plot_options(4)

    if training == 'FeatureExtraction':
        # Top Right
        plot_options[0][1]['x_min'] = 1200
        plot_options[0][1]['x_max'] = 1500
        plot_options[0][1]['y_min'] = 0.7
        plot_options[0][1]['y_max'] = 0.95

        # Bottom Right
        plot_options[1][1]['x_min'] = 1200
        plot_options[1][1]['x_max'] = 1500
        plot_options[1][1]['y_min'] = 0.45
        plot_options[1][1]['y_max'] = 0.75
    elif training == 'FineTuned':
        # Top Right
        plot_options[0][1]['x_min'] = 0
        plot_options[0][1]['x_max'] = 200
        plot_options[0][1]['y_min'] = 0.8
        plot_options[0][1]['y_max'] = 1.05

        # Bottom Right
        plot_options[1][1]['x_min'] = -10
        plot_options[1][1]['x_max'] = 410
        plot_options[1][1]['y_min'] = 0.3
        plot_options[1][1]['y_max'] = 0.9
    elif training == 'FromScratch':
        # Top Right
        plot_options[0][1]['x_min'] = 900
        plot_options[0][1]['x_max'] = 1500
        plot_options[0][1]['y_min'] = 0.97
        plot_options[0][1]['y_max'] = 1.01

        # Bottom Left
        plot_options[1][0]['y_max'] = 0.8

        # Bottom Right
        plot_options[1][1]['x_min'] = 600
        plot_options[1][1]['x_max'] = 1200
        plot_options[1][1]['y_min'] = 0.3
        plot_options[1][1]['y_max'] = 0.8

    fig, axs = plt.subplots(2, 2)

    for d in data.values():
        draw_plot(axs[0, 0], d[METRIC_TRAINING])
        draw_plot(axs[0, 1], d[METRIC_TRAINING])
        draw_plot(axs[1, 0], d[METRIC_VALIDATION])
        draw_plot(axs[1, 1], d[METRIC_VALIDATION])

    for y in range(2):
        for x in range(2):
            annotate_subplot(axs[y, x], **plot_options[y][x])

    finalize_figure()


def plot_architecture_comparisons():
    plot_architecture_comparison('FromScratch')
    plot_architecture_comparison('FineTuned')
    plot_architecture_comparison('FeatureExtraction')


# noinspection PyTypeChecker
def plot_class_removal_comparison():
    directory = os.path.join(ROOT_DIR, 'classes')

    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'original': get_data(os.path.join(directory, 'all'), metrics=metrics, color='b', label='Original'),
        'without noisy classes': get_data(os.path.join(directory, 'noisy removed'), metrics=metrics, color='g',
                                          label='Noisy Classes Removed')
    }

    print_data(data)

    plot_options = get_standard_plot_options(4)

    # Top Right
    plot_options[0][1]['x_min'] = 1000
    plot_options[0][1]['x_max'] = 1500
    plot_options[0][1]['y_min'] = 0.75
    plot_options[0][1]['y_max'] = 1.05

    # Bottom Right
    plot_options[1][1]['x_min'] = 1000
    plot_options[1][1]['x_max'] = 1500
    plot_options[1][1]['y_min'] = 0.3
    plot_options[1][1]['y_max'] = 0.8

    fig, axs = plt.subplots(2, 2)

    for d in data.values():
        draw_plot(axs[0, 0], d[METRIC_TRAINING])
        draw_plot(axs[0, 1], d[METRIC_TRAINING])
        draw_plot(axs[1, 0], d[METRIC_VALIDATION])
        draw_plot(axs[1, 1], d[METRIC_VALIDATION])

    for y in range(2):
        for x in range(2):
            annotate_subplot(axs[y, x], **plot_options[y][x])

    finalize_figure()


# noinspection PyTypeChecker
def plot_improvement_comparison():
    directory = os.path.join(ROOT_DIR, 'architecture_improvements')

    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'original': get_data(os.path.join(directory, 'original'), metrics=metrics, color='b', label='Original'),
        'update ResNet': get_data(os.path.join(directory, 'ResNetModule update'), metrics=metrics, color='r',
                                  label='ResNet18 Modules'),
        'add ResNet': get_data(os.path.join(directory, 'additional ResNetModule'), metrics=metrics, color='g',
                               label='Additional ResNet Module')
    }

    print_data(data)

    plot_options = get_standard_plot_options(4)

    # Top Right
    plot_options[0][1]['x_min'] = 1000
    plot_options[0][1]['x_max'] = 1500
    plot_options[0][1]['y_min'] = 0.9
    plot_options[0][1]['y_max'] = 1.05

    # Bottom Right
    plot_options[1][1]['x_min'] = 1000
    plot_options[1][1]['x_max'] = 1500
    plot_options[1][1]['y_min'] = 0.55
    plot_options[1][1]['y_max'] = 0.85

    fig, axs = plt.subplots(2, 2)

    for d in data.values():
        draw_plot(axs[0, 0], d[METRIC_TRAINING])
        draw_plot(axs[0, 1], d[METRIC_TRAINING])
        draw_plot(axs[1, 0], d[METRIC_VALIDATION])
        draw_plot(axs[1, 1], d[METRIC_VALIDATION])

    for y in range(2):
        for x in range(2):
            annotate_subplot(axs[y, x], **plot_options[y][x])

    finalize_figure()


# noinspection PyTypeChecker
def plot_transfer_learning_comparison():
    directory = os.path.join('models', 'final')

    metrics = [{
        'name': METRIC_TRAINING
    }, {
        'name': METRIC_VALIDATION
    }]

    data = {
        'target': get_data(os.path.join(directory, 'target'), metrics=metrics, color='b', label='Target from Scratch'),
        'transfer fe': get_data(os.path.join(directory, 'transfer_fe'), metrics=metrics, color='r', label='Feature Extraction'),
        'transfer ft': get_data(os.path.join(directory, 'transfer_ft'), metrics=metrics, color='g', label='Fine-Tuning')
    }

    print_data(data)

    plot_options = get_standard_plot_options(4)

    # Top Right
    plot_options[0][1]['x_min'] = 0
    plot_options[0][1]['x_max'] = 500
    plot_options[0][1]['y_min'] = 0.75
    plot_options[0][1]['y_max'] = 1.05

    # Bottom Right
    plot_options[1][1]['x_min'] = 0
    plot_options[1][1]['x_max'] = 500
    plot_options[1][1]['y_min'] = 0.25
    plot_options[1][1]['y_max'] = 0.8

    fig, axs = plt.subplots(2, 2)

    for d in data.values():
        draw_plot(axs[0, 0], d[METRIC_TRAINING])
        draw_plot(axs[0, 1], d[METRIC_TRAINING])
        draw_plot(axs[1, 0], d[METRIC_VALIDATION])
        draw_plot(axs[1, 1], d[METRIC_VALIDATION])

    for y in range(2):
        for x in range(2):
            annotate_subplot(axs[y, x], **plot_options[y][x])

    finalize_figure()


def main():
    plot_hyper_parameter_comparisons()
    plot_augmentation_comparisons()
    plot_architecture_comparisons()
    plot_class_removal_comparison()
    plot_improvement_comparison()
    plot_transfer_learning_comparison()


if __name__ == '__main__':
    main()
