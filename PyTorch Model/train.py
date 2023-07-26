import argparse
from model_functions import train_model

# !tensorboard --logdir=runs_KingsCollege
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='./datasets/KingsCollege')
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--model', type=str, default='resnet', choices=['googlenet', 'resnet'])
    parser.add_argument('--pretrained_model', type=str, default=None, help='model path: epoch')
    parser.add_argument('--learn_beta', type=bool, default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='range 0.0 to 1.0')
    parser.add_argument('--fixed_weight', type=bool, default=False)

    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_epochs_decay', type=int, default=50)

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--model_save_step', type=int, default=50)

    user_args = parser.parse_args()
    train_model(user_args)
