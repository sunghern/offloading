# -*- coding=utf-8 -*-

import os
import torch
from tqdm import tqdm
from loguru import logger
from models import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Pytorch FBS training')
cuda1 = torch.device('cuda:1')

def watch_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError('found NaN: ' + str(x))

def main():
    name = 'resnet20-imagenette2'
    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', name + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)
    net = resnet20(num_classes=10).cuda(cuda1)
    #train_loader = imagenette2('train')
    #val_loader = imagenette2('val')
    train_loader, val_loader = data_loader('.', dataset='cifar10', batch_size=64, workers=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.2)
    loss_function = torch.nn.CrossEntropyLoss()
    best_accuracy = 0

    checkpoint = torch.load("ckpts/0.6-latest.pth")
    net.load_state_dict(checkpoint)

    '''
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.weight.mul_(6/10)
                m.bias.mul_(6/10)
    '''

    with tqdm(val_loader) as valid_tqdm:

        print("validation start")
        for pruning_rate in [1.0]:
            logger.info('set validation pruning rate = %.1f' % pruning_rate)
            for m in net.modules():
                if hasattr(m, 'rate'):
                    m.rate = pruning_rate

        print("validation pr:  ", pruning_rate)

        #valid_tqdm.set_description_str('{:03d} valid'.format(epoch))
        net.eval()
        meter = AccuracyMeter(topk=(1, 5))
        with torch.no_grad():
            for images, labels in valid_tqdm:
                images = images.cuda(cuda1)
                labels = labels.cuda(cuda1)
                output = net(images)
                watch_nan(output)
                meter.update(output, labels)
                valid_tqdm.set_postfix(meter.get())
        logger.info('valid result: {}'.format(meter.get()))

if __name__ == '__main__':
    main()
