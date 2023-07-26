import argparse
from model_functions import test_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='resnet', choices=['googlenet', 'resnet'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default='./datasets/KingCollege')

    user_args = parser.parse_args()
    test_model(user_args)
