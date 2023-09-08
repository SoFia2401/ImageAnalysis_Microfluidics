## Employ mlflow to log the training process disable if mlflow is not installed

import mlflow
import argparse
import os
import copy

import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr, calc_ssim
import torchmetrics
import matplotlib.pyplot as plt


from mlflow.models.signature import infer_signature
import mlflow.sklearn
from urllib.parse import urlparse
import numpy as np


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.font_manager as font_manager



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()
    


    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device " + 'cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    train_dataset = EvalDataset(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    # Keep track of loss ssim and psnr during training
    train_loss = AverageMeter()
    train_psnr = AverageMeter()
    train_ssim = AverageMeter()
    val_psnr = AverageMeter()
    val_ssim = AverageMeter()
    val_loss = AverageMeter()
    
    mlflow.set_experiment("Experiment 3")
    mlflow.autolog()
    
    psnr_values = []

    with mlflow.start_run(nested=True, run_name = ("x" + str(args.scale) + "_srcnn_batchSize_" + str(args.batch_size) + "_" + args.outputs_dir)):
        for epoch in range(args.num_epochs):
            train_loss.reset()
            train_psnr.reset()
            train_ssim.reset()
            val_loss.reset()
            val_psnr.reset()
            val_loss.reset()
            model.train()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
                t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                for data in train_dataloader:
                    inputs, labels = data

                    inputs = inputs.to(device).float()
                    labels = labels.to(device).float()

                    preds = model(inputs)

                    loss = criterion(preds, labels)

                    with torch.no_grad():
                        train_loss.update(loss.item(), len(inputs))
                        train_psnr.update_psnr(calc_psnr(preds, labels), len(inputs))
                        train_ssim.update_ssim(calc_ssim(preds, labels), len(inputs))

                        t.set_postfix(loss='{:.8f}'.format(train_loss.avg))
                        t.update(len(inputs))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Convert tensors to NumPy arrays
                    labels_np = labels.cpu().detach().numpy()
                    preds_np = preds.cpu().detach().numpy()

                    # Create a dictionary of input data
                    data = {'labels': labels_np, 'preds': preds_np}

                    # Infer the signature from the input data dictionary
                    signature = infer_signature(data)
                    
                    
                
            train_loss.update_epoch(train_loss.avg)
            train_psnr.update_epoch(train_psnr.avg)
            train_ssim.update_epoch(train_ssim.avg)

           # mlflow.pytorch.log_model(model, "model", signature=signature)

            
            mlflow.log_metric("train loss", train_loss.per_epoch[epoch])
            mlflow.log_metric("train psnr", train_psnr.per_epoch[epoch])
            mlflow.log_metric("train ssim", train_ssim.per_epoch[epoch])
            torch.save(model.state_dict(), os.path.join(args.outputs_dir, 'epoch_{}.pth'.format(epoch)))

            model.eval()

            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                    loss = criterion(preds, labels)
                    val_loss.update(loss.item(), len(inputs))

                    val_psnr.update_psnr(calc_psnr(preds, labels), len(inputs))
                    #(calc_psnr(inputs, labels))
                    val_ssim.update_ssim(calc_ssim(preds, labels), len(inputs))
                    
            print(calc_psnr(inputs, labels))
            print('Val loss: {:.8f}'.format(val_loss.avg))
            val_loss.update_epoch(val_loss.avg)
            val_psnr.update_epoch(val_psnr.avg)
            val_ssim.update_epoch(val_ssim.avg)

            mlflow.log_metric("val loss", val_loss.per_epoch[epoch])
            mlflow.log_metric("val psnr", val_psnr.per_epoch[epoch])
            mlflow.log_metric("val ssim", val_ssim.per_epoch[epoch])

            fontSize = 24
            #Define the font properties
            font = font_manager.FontProperties(family='Times New Roman', size=fontSize)


            # loss plots
            plt.figure(figsize=(10, 7))
            plt.plot(train_loss.per_epoch, color='orange', label='Train loss', linewidth=3)
            plt.plot(val_loss.per_epoch, color='red', label='Validation loss', linewidth=3)
            plt.xlabel('Epochs', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.ylabel('Loss', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.xticks(fontsize=fontSize, fontproperties=font)  # Increase x-axis tick font size
            plt.yticks(fontsize=fontSize, fontproperties=font)  # Increase y-axis tick font size
            plt.legend(prop=font)
            plt.savefig(args.outputs_dir + '/loss.pdf', format='pdf')
            plt.close()
            
            # psnr plots
            plt.figure(figsize=(10, 7))
            plt.plot(train_psnr.per_epoch, color='green', label='Train PSNR dB', linewidth=3)
            plt.plot(val_psnr.per_epoch, color='blue', label='Validation PSNR dB', linewidth=3)
            plt.xlabel('Epochs', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.ylabel('PSNR (dB)', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.xticks(fontsize=fontSize, fontproperties=font)  # Increase x-axis tick font size
            plt.yticks(fontsize=fontSize, fontproperties=font)  # Increase y-axis tick font size
            plt.legend(prop=font)
            plt.savefig(args.outputs_dir + '/psnr.pdf', format='pdf')
            plt.close()
            
            # ssim plots
            plt.figure(figsize=(10, 7))
            plt.plot(train_ssim.per_epoch, color='green', label='Train SSIM', linewidth=3)
            plt.plot(val_ssim.per_epoch, color='blue', label='Validation SSIM', linewidth=3)
            plt.xlabel('Epochs', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.ylabel('SSIM', fontproperties=font, fontsize=fontSize)  # Set fontsize to 12
            plt.xticks(fontsize=fontSize, fontproperties=font)  # Increase x-axis tick font size
            plt.yticks(fontsize=fontSize, fontproperties=font)  # Increase y-axis tick font size
            plt.legend(prop=font)
            plt.savefig(args.outputs_dir + '/ssim.pdf', format='pdf')
            plt.close()


            psnr_values.append(val_psnr.per_epoch[epoch])

            if val_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = val_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

            print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
            
        print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(args.outputs_dir, 'best.pth'))