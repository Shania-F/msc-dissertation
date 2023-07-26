import os
import time
import numpy as np
from tensorboardX import SummaryWriter

from torch.backends import cudnn
# enables auto-tuning of the convolution algorithms
# for improved performance on the available GPU
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader import get_dataloader
import models
from loss import GeoPoseLoss
from pose_utils import *


def train_model(save_name='KingsCollege'):
    # config = user_args
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # set seed

    # DATA
    dataset_path = './datasets/KingsCollege'

    data_name = dataset_path.split('/')[-1]
    model_save_path = 'models_%s' % data_name
    summary_save_path = 'summary_%s' % data_name

    # MODEL
    model = models.ResNetVLAD()
    model.to(device)

    if config.pretrained_model:  # either none or the path and epoch to start from
        model_path = self.model_save_path + '/%s_net.pth' % self.config.pretrained_model
        self.model.load_state_dict(torch.load(model_path))
        print('Load pretrained network: ', model_path)

    # LOSS AND OPTIMISER
    criterion = GeoPoseLoss(config.learn_beta)
    criterion.to(device)

    if config.learn_beta:
        optimizer = optim.Adam([{'params': model.parameters()},
                                {'params': [criterion.sx, criterion.sq]}],
                               lr=config.lr,
                               weight_decay=0.0005)
    else:
        optimizer = optim.Adam(model.parameters(),
                               lr=config.lr,
                               weight_decay=0.0005)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.num_epochs_decay, gamma=0.1)

    # TRAINING
    num_epochs = config.num_epochs

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # tensorboard
    if not os.path.exists(summary_save_path):
        os.makedirs(summary_save_path)
    writer = SummaryWriter(log_dir=summary_save_path)

    start_time = time.time()
    n_iter = 0  # no of batches per epoch
    start_epoch = 0

    previous_centroids = model.net_vlad.centroids.data.clone()
    print("OLD CENTROIDS: ", previous_centroids)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        error_train = []
        error_val = []

        # TRAIN
        # scheduler.step()
        model.train()

        train_loader = get_dataloader(train=True)

        for i, (inputs, poses) in enumerate(train_loader):
            inputs = inputs.to(device)
            # print("INPUT", inputs.shape)
            poses = poses.to(device)
            # print("POSES", poses)

            # Zero the parameter gradient
            optimizer.zero_grad()

            # forward
            pos_out, ori_out = model(inputs)
            # print("PRED POSE", pos_out, "PRED ORI", ori_out)

            pos_true = poses[:, :3]
            ori_true = poses[:, 3:]
            # print("ACT POSE", pos_true, "ACT ORI", ori_true)

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss, loss_pos_print, loss_ori_print = criterion(pos_out, ori_out, pos_true, ori_true)
            # print("LOSS", loss.item())
            loss_print = loss.item()

            error_train.append(loss_print)
            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)

            loss.backward()
            optimizer.step()
            n_iter += 1

            print('batch#{} train Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, loss_print,
                                                                                                      loss_pos_print,
                                                                                                      loss_ori_print))

        # ALL BATCHES DONE TRAINING
        with torch.no_grad():
            new_centroids = model.net_vlad.centroids.data
            print("NEW CENTROIDS :", new_centroids)
            # calculate l2 norm and take mean of distances between all old-new centroid pairs
            centroid_distance = torch.norm(new_centroids - previous_centroids, dim=1).mean()
            print(f"Centroid Distance from Previous Epoch: {centroid_distance.item()}")

        # Update the previous centroids to the new centroids for the next epoch
        previous_centroids = model.net_vlad.centroids.data.clone()

        # VALIDATION
        model.eval()
        val_loader = get_dataloader(train=False)

        for i, (inputs, poses) in enumerate(val_loader):
            inputs = inputs.to(device)
            poses = poses.to(device)

            # Zero the parameter gradient
            optimizer.zero_grad()

            # forward
            pos_out, ori_out = model(inputs)

            pos_true = poses[:, :3]
            ori_true = poses[:, 3:]

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss, loss_pos_print, loss_ori_print = criterion(pos_out, ori_out, pos_true, ori_true)
            loss_print = loss.item()

            error_val.append(loss_print)
            print('batch#{} val Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, loss_print,
                                                                                                    loss_pos_print,
                                                                                                    loss_ori_print))

        # END OF EPOCH
        error_train_loss = np.median(error_train)
        error_val_loss = np.median(error_val)

        print('Train and Validation error {} / {}'.format(error_train_loss, error_val_loss))
        print('=' * 40)
        print('=' * 40)

        writer.add_scalars('loss/trainval', {'train': error_train_loss,
                                             'val': error_val_loss}, epoch)

        # SAVE MODEL ...every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_filename = model_save_path + '/model.pth'
            torch.save(model.cpu().state_dict(), save_filename)
            if torch.cuda.is_available():
                model.to(device)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# validation step
# this is the test step, but we get to adjust hyper parameters to improve this
def val_model():
    pass


def test_model(save_name):
    model_save_path = 'models_%s' % save_name
    summary_save_path = 'summary_%s' % save_name
    # f = open(summary_save_path + '/test_result.csv', 'w+')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # model = models.ResNet(fixed_weight=True)
    model = models.ResNetV(fixed_weight=True)
    model.to(device)

    model.eval()
    test_model_path = model_save_path + '/model.pth'

    print('Loading pretrained model: ', test_model_path)
    model.load_state_dict(torch.load(test_model_path))

    total_pos_loss = 0
    total_ori_loss = 0
    # arrays to store individual losses
    pos_loss_arr = []
    ori_loss_arr = []
    true_pose_list = []
    estim_pose_list = []

    test_loader = get_dataloader(train=False)  # BATCH SIZE IS 1
    for i, (inputs, poses) in enumerate(test_loader):
        print(i)

        inputs = inputs.to(device)
        pos_out, ori_out = model(inputs)

        # Remove extra [batch of 1] dim from pos_out by calling squeeze(0).
        # Then, detach the tensor from the computation graph using detach()
        pos_out = pos_out.squeeze(0).detach().cpu().numpy()
        ori_out = F.normalize(ori_out, p=2, dim=1)
        ori_out = quat_to_euler(ori_out.squeeze(0).detach().cpu().numpy())
        print('pos out', pos_out)
        print('ori_out', ori_out)

        pos_true = poses[:, :3].squeeze(0).numpy()
        ori_true = poses[:, 3:].squeeze(0).numpy()

        ori_true = quat_to_euler(ori_true)
        print('pos true', pos_true)
        print('ori true', ori_true)
        # l2 norm
        loss_pos_print = array_dist(pos_out, pos_true)
        loss_ori_print = array_dist(ori_out, ori_true)

        true_pose_list.append(np.hstack((pos_true, ori_true)))
        estim_pose_list.append(np.hstack((pos_out, ori_out)))

        total_pos_loss += loss_pos_print
        total_ori_loss += loss_ori_print

        pos_loss_arr.append(loss_pos_print)
        ori_loss_arr.append(loss_ori_print)

    position_error = np.median(pos_loss_arr)
    rotation_error = np.median(ori_loss_arr)

    print('=' * 20)
    print('Overall median pose error {:.3f}m / {:.3f}rad'.format(position_error, rotation_error))
    print('Overall average pose error {:.3f}m / {:.3f}rad'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
    # f.close()

    f_true = summary_save_path + '/pose_true.csv'
    f_estim = summary_save_path + '/pose_estim.csv'
    np.savetxt(f_true, true_pose_list, delimiter=',')
    np.savetxt(f_estim, estim_pose_list, delimiter=',')


if __name__ == '__main__':
    # train_model(save_name='KingsCollege-ResNet')
    # test_model(save_name='KingsCollege-ResNet')

    # train_model(save_name='KingsCollege-GNet')
    train_model(save_name='KingsCollege-ResNet-V')
    # test_model(save_name='KingsCollege-ResNet-V')

