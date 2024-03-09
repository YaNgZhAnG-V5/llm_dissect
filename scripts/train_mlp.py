import os.path as osp
from argparse import ArgumentParser
from datetime import datetime

import mmengine
import torch
import torch.nn as nn
import torchvision.transforms as T
from mmengine.runner import set_random_seed
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from dissect.models import MLP


def parse_args():
    parser = ArgumentParser('Train MLP')
    parser.add_argument(
        '--work-dir', '-w', default='workdirs/debug/', help='Working directory to save the output files.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID.')

    return parser.parse_args()


def main():
    set_random_seed(42)
    args = parse_args()
    work_dir = args.work_dir
    mmengine.mkdir_or_exist(work_dir)
    ckpt_dir = osp.join(work_dir, 'ckpts')
    mmengine.mkdir_or_exist(ckpt_dir)

    logger = mmengine.MMLogger.get_instance(
        name='dissect',
        logger_name='dissect',
        log_file=osp.join(work_dir, f'{datetime.now().strftime("%y%m%d_%H%M")}.log'))

    device = torch.device(f'cuda:{args.gpu_id}')

    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])
    train_set = MNIST('./data/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=256, num_workers=2, pin_memory=True, shuffle=True)
    test_set = MNIST('./data/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, num_workers=2, pin_memory=True, shuffle=False)

    model = MLP([784, 1024, 1024, 512, 256, 10]).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        model.train()
        for batch_idx, (image, target) in enumerate(train_loader):
            optimizer.zero_grad()
            image, target = image.to(device), target.to(device)
            output = model(image, flatten_start_dim=1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(f'Epoch: {epoch + 1} [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.6f}')

        num_correct = 0
        model.eval()
        with torch.no_grad():
            for image, target in test_loader:
                image, target = image.to(device), target.to(device)
                pred = model(image, flatten_start_dim=1).argmax(-1)
                num_correct += (pred == target).sum().item()

        acc = num_correct / len(test_set)
        logger.info(f'Epoch: {epoch + 1} Test Accuracy: {acc:.4f}')

    save_path = osp.join(ckpt_dir, f'trained_model.pth')
    torch.save(model.state_dict(), save_path)

    logger.info(f'Checkpoints are saved to: {ckpt_dir}')


if __name__ == '__main__':
    main()
