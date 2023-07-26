import os
import time
import numpy as np
from tensorboardX import SummaryWriter

# enables auto-tuning of the convolution algorithms
# for improved performance on the available GPU
from torch.backends import cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from data_loader import get_dataloader
import models
from loss import GeoPoseLoss
from pose_utils import *


def train_model(config):
    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    # TODO set seed
    # TODO print out config

    # DATA
    # dataset_path = './datasets/KingsCollege'
    data_name = config.dataset_path.split('/')[-1]
    model_save_path = 'models_%s' % data_name
    tb_save_path = 'runs_%s' % data_name

    train_loader = get_dataloader(dataset_path=config.dataset_path, mode='train',
                                  model=config.model, batch_size=config.batch_size)
    val_loader = get_dataloader(dataset_path=config.dataset_path, mode='val',
                                model=config.model, batch_size=config.batch_size)

    # MODEL
    if config.model == 'googlenet':
        model = models.GoogleNet(fixed_weight=config.fixed_weight, dropout_rate=config.dropout_rate)
    else:
        model = models.ResNet(fixed_weight=config.fixed_weight, dropout_rate=config.dropout_rate)
    model.to(device)

    if config.pretrained_model:  # TODO either none or the path and epoch to start from
        model_path = '/%s_net.pth' % config.pretrained_model
        model.load_state_dict(torch.load(model_path))
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

    # LR is decayed by the value of gamma
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.num_epochs_decay, gamma=0.1)

    # TRAINING
    num_epochs = config.num_epochs

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # tensorboard
    if not os.path.exists(tb_save_path):
        os.makedirs(tb_save_path)
    writer = SummaryWriter(log_dir=tb_save_path)

    start_time = time.time()
    n_iter = 0  # total no of batches over all epochs, useful for logger
    start_epoch = 0
    if config.pretrained_model:
        start_epoch = int(config.pretrained_model)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 20)

        error_train = []
        error_val = []

        model.train()

        for i, (inputs, poses) in enumerate(train_loader):

            inputs = inputs.to(device)
            # print("INPUT", inputs.shape)
            poses = poses.to(device)
            # print("POSES", poses)

            # Zero the parameter gradient
            optimizer.zero_grad()

            # FORWARD
            pos_out, ori_out = model(inputs)
            # print("PRED POSE", pos_out, "PRED ORI", ori_out)

            # COMPUTE LOSS
            pos_true = poses[:, :3]
            ori_true = poses[:, 3:]
            # print("ACT POSE", pos_true, "ACT ORI", ori_true)

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss, loss_pos_print, loss_ori_print = criterion(pos_out, ori_out, pos_true, ori_true)
            # print("LOSS", loss.item())
            loss_print = loss.item()

            # loss = loss_pos + beta * loss_ori

            # TODO use config.log_step
            error_train.append(loss_print)
            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
            if config.learn_beta:
                writer.add_scalar('param/sx', criterion.sx.item(), n_iter)
                writer.add_scalar('param/sq', criterion.sq.item(), n_iter)

            # BACKWARD
            loss.backward()
            optimizer.step()
            scheduler.step()
            n_iter += 1

            print('batch#{} train Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, loss_print,
                                                                                               loss_pos_print,
                                                                                               loss_ori_print))

        # ALL BATCHES DONE TRAINING
        # VALIDATION
        model.eval()

        for i, (inputs, poses) in enumerate(val_loader):

            inputs = inputs.to(device)
            poses = poses.to(device)

            # Zero the parameter gradient
            optimizer.zero_grad()

            # FORWARD
            pos_out, ori_out = model(inputs)

            # COMPUTE LOSS
            pos_true = poses[:, :3]
            ori_true = poses[:, 3:]

            ori_out = F.normalize(ori_out, p=2, dim=1)
            ori_true = F.normalize(ori_true, p=2, dim=1)

            loss, loss_pos_print, loss_ori_print = criterion(pos_out, ori_out, pos_true, ori_true)
            loss_print = loss.item()

            error_val.append(loss_print)
            # TODO log val loss
            print('batch#{} val Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, loss_print,
                                                                                              loss_pos_print,
                                                                                               loss_ori_print))

        # END OF EPOCH
        error_train_loss = np.median(error_train)
        error_val_loss = np.median(error_val)

        print('Overall Train and Validation loss for epoch {} / {}'.format(error_train_loss, error_val_loss))
        print('=' * 40)
        print('=' * 40)

        writer.add_scalars('loss/trainval', {'train': error_train_loss, 'val': error_val_loss}, epoch+1)

        if (epoch + 1) % config.model_save_step == 0:
            # save_filename = model_save_path + '/%s_net.pth' % epoch
            save_filename = model_save_path + '/posenet.pth'
            torch.save(model.cpu().state_dict(), save_filename)
            if torch.cuda.is_available():
                model.to(device)

        # SAVE BEST while training
        # if error_train_loss < best_train_loss:
        #     best_train_loss = error_train_loss
        #     best_train_model = epoch
        # if error_val_loss < best_val_loss:
        #     best_val_loss = error_val_loss
        #     best_val_model = epoch
        #     save_filename = self.model_save_path + '/best_net.pth'
        #     torch.save(self.model.cpu().state_dict(), save_filename)
        #     if torch.cuda.is_available():
        #         self.model.to(self.device)

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# validation step
# this is the test step, but we get to adjust hyper parameters to improve this
def val_model():
    pass


def test_model(config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)

    # LOAD THE MODEL
    data_name = config.dataset_path.split('/')[-1]
    model_save_path = 'models_%s' % data_name
    tb_save_path = 'runs_%s' % data_name
    # f = open(tb_save_path + '/test_result.csv', 'w+')

    if config.pretrained_model:
        model_save_path = config.pretrained_model

    if config.model == 'googlenet':
        model = models.GoogleNet(fixed_weight=config.fixed_weight, dropout_rate=config.dropout_rate)
    else:
        model = models.ResNet(fixed_weight=config.fixed_weight, dropout_rate=config.dropout_rate)
    model.to(device)

    print('Loading pretrained model: ', model_save_path)
    model.load_state_dict(torch.load(model_save_path))

    model.eval()

    total_pos_loss = 0
    total_ori_loss = 0
    # arrays to store individual losses
    pos_loss_arr = []
    ori_loss_arr = []
    true_pose_list = []
    estim_pose_list = []

    test_loader = get_dataloader(dataset_path=config.dataset_path, mode='test')  # BATCH SIZE IS 1
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
        print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))

    position_error = np.median(pos_loss_arr)
    rotation_error = np.median(ori_loss_arr)

    print('=' * 20)
    print('Overall median pose error {:.3f}m / {:.3f}rad'.format(position_error, rotation_error))
    print('Overall average pose error {:.3f}m / {:.3f}rad'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
    # f.close()

    f_true = tb_save_path + '/pose_true.csv'
    f_estim = tb_save_path + '/pose_estim.csv'
    np.savetxt(f_true, true_pose_list, delimiter=',')
    np.savetxt(f_estim, estim_pose_list, delimiter=',')


if __name__ == '__main__':
    # train_model(save_name='KingsCollege-ResNet')
    # test_model(save_name='KingsCollege-ResNet')

    # train_model(save_name='KingsCollege-GNet')
    train_model(save_name='KingsCollege-ResNet-V')
    # test_model(save_name='KingsCollege-ResNet-V')

