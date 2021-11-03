# -*- coding=utf-8 -*-

import os
import torch
from tqdm import tqdm
from loguru import logger
from models import *
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Pytorch FBS training')


def watch_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError('found NaN: ' + str(x))

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss

def main():
    name = 'resnet34-imagenette2'
    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', name + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)
    net = resnet34(num_classes=10)

    pruning_rate_list = [1.0, 0.8, 0.6]
    for m in net.modules():
        if isinstance(m, SwitchableBatchNorm2d):
            m.make_list(pruning_rate_list)
    net = net.cuda()

    #train_loader = imagenette2('train')
    #val_loader = imagenette2('val')
    train_loader, val_loader = data_loader('.', dataset='cifar10', batch_size=64, workers=8)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.2)
    loss_function = torch.nn.CrossEntropyLoss()
    soft_loss_function = CrossEntropyLossSoft()
    best_accuracy = 0
    for epoch in range(100): #epoch = 100
        with tqdm(train_loader) as train_tqdm: #open file - train file
            train_tqdm.set_description_str('{:03d} train'.format(epoch))
            net.train()
            meter = AccuracyMeter(topk=(1, 5))

            for images, labels in train_tqdm:
                images = images.cuda() #move image to GPU
                labels = labels.cuda() #move image to GPU
                #loss = 0 #loss initialize

                optimizer.zero_grad()
                for pruning_rate in pruning_rate_list: 
                    #logger.info('set pruning rate = %.1f' % pruning_rate)
                    for m in net.modules():#set pruning rate
                        if hasattr(m, 'rate'): #check m if 'rate' is existing
                            m.rate = pruning_rate

                    output = net(images)
                    watch_nan(output)
                    meter.update(output, labels)
                    train_tqdm.set_postfix(meter.get())
                    if pruning_rate == 1.0:
                        loss1 = loss_function(output, labels)
                        soft_target = torch.nn.functional.softmax(output, dim=1)
                    else:
                        loss1 = torch.mean(soft_loss_function(output, soft_target.detach()))
                    loss2 = 0

                    if pruning_rate != 1.0:
                        for m in net.modules():
                            if hasattr(m, 'loss') and m.loss is not None:
                                loss2 += m.loss

                    loss = loss1 + 1e-8 * loss2
                    loss.backward()
                #loss.backward()
                optimizer.step()

            logger.info('{:03d} train result: {}'.format(epoch, meter.get()))

        with tqdm(val_loader) as valid_tqdm:

            for pruning_rate in [1.0]:
                for m in net.modules():
                    if hasattr(m, 'rate'):
                        m.rate = pruning_rate

            valid_tqdm.set_description_str('{:03d} valid'.format(epoch))
            net.eval()
            meter = AccuracyMeter(topk=(1, 5))
            with torch.no_grad():
                for images, labels in valid_tqdm:
                    images = images.cuda()
                    labels = labels.cuda()
                    output = net(images)
                    watch_nan(output)
                    meter.update(output, labels)
                    valid_tqdm.set_postfix(meter.get())
            logger.info('{:03d} valid result: {}'.format(epoch, meter.get()))
        if (epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), 'ckpts/%.1f-latest.pth' % pruning_rate)
            logger.info('saved to ckpts/%.1f-latest.pth' % pruning_rate)
        if best_accuracy < meter.top():
            best_accuracy = meter.top()
            torch.save(net.state_dict(), 'ckpts/%.1f-best.pth' % pruning_rate)
            logger.info('saved to ckpts/%.1f-best.pth' % pruning_rate)
        scheduler.step()

if __name__ == '__main__':
    main()
