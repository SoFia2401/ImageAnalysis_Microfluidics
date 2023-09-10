import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
import torch
from torch.utils.data import DataLoader
from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option

from data.select_dataset import define_Dataset
from models.select_model import define_Model

from mlflow.models.signature import infer_signature
import mlflow.pytorch

from utils_metrics import AverageMeter, calc_psnr, calc_ssim

def main(json_path='options/train_dncnn.json'):
    
    
    mlflow.set_experiment("Experiment 5")
    mlflow.autolog()
    
    with mlflow.start_run(nested=True):
  
        parser = argparse.ArgumentParser()
        parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    
        opt = option.parse(parser.parse_args().opt, is_train=True)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))
    
        init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
        opt['path']['pretrained_netG'] = init_path_G
        current_step = init_iter
    
        border = 0
        option.save(opt)
    
        opt = option.dict_to_nonedict(opt)
      
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        logger.info(option.dict2str(opt))
    
        seed = opt['train']['manual_seed']
        if seed is None:
            seed = random.randint(1, 10000)
        logger.info('Random seed: {}'.format(seed))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
        dataset_type = opt['datasets']['train']['dataset_type']
        for phase, dataset_opt in opt['datasets'].items():
            if phase == 'train':
                train_set = define_Dataset(dataset_opt)
                train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)
            elif phase == 'test':
                test_set = define_Dataset(dataset_opt)
                test_loader = DataLoader(test_set, batch_size=1,
                                         shuffle=False, num_workers=1,
                                         drop_last=False, pin_memory=True)
            else:
                raise NotImplementedError("Phase [%s] is not recognized." % phase)
    
    
        model = define_Model(opt)
    
        if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
            logger.info('^_^ -----merging bnorm----- ^_^')
            model.merge_bnorm_test()
    
        logger.info(model.info_network())
        model.init_train()
        logger.info(model.info_params())
    
        train_loss = AverageMeter()
        train_psnr = AverageMeter()
        train_ssim = AverageMeter()
        val_psnr = AverageMeter()
        val_ssim = AverageMeter()
        val_loss = AverageMeter()
    
        for epoch in range(1200):  # keep running
        
            train_loss.reset()
            train_psnr.reset()
            train_ssim.reset()
            val_loss.reset()
            val_psnr.reset()
            val_ssim.reset()
            
            for i, train_data in enumerate(train_loader):
    
                current_step += 1
    
                if dataset_type == 'dnpatch' and current_step % 20000 == 0:  # for 'train400'
                    train_loader.dataset.update_data()
    

                model.feed_data(train_data)
                
                model.update_learning_rate(current_step)
    

                model.optimize_parameters(current_step)
                
                
                
                visuals = model.current_visuals()
                E_img = util.tensor2uint(visuals['E'])
                H_img = util.tensor2uint(visuals['H'])
                L_img = util.tensor2uint(visuals['L'])
                
                logs = model.current_log()  # such as loss
                
                train_loss.update(logs['G_loss'], len(train_data['H']))
                train_psnr.update_psnr(calc_psnr(E_img, H_img), len(E_img))
                train_ssim.update_ssim(calc_ssim(E_img, H_img), len(E_img))
                
    
                # -------------------------------
                # merge bnorm
                # -------------------------------
                if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                    logger.info('^_^ -----merging bnorm----- ^_^')
                    model.merge_bnorm_train()
                    model.print_network()
    
                # -------------------------------
                # 4) training information
                # -------------------------------
                if current_step % opt['train']['checkpoint_print'] == 0:
                    logs = model.current_log()  # such as loss
                    message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                    
                    train_loss.update_epoch(train_loss.avg)
                    train_psnr.update_epoch(train_psnr.avg)
                    train_ssim.update_epoch(train_ssim.avg)
                    #mlflow.pytorch.log_model(model, "model")
    
                    mlflow.log_metric("train loss", train_loss.per_epoch[epoch])
                    mlflow.log_metric("train psnr", train_psnr.per_epoch[epoch])
                    mlflow.log_metric("train ssim", train_ssim.per_epoch[epoch])
                    
                    
                    for k, v in logs.items():  # merge log information into message
                        message += '{:s}: {:.3e} '.format(k, v)
                    logger.info(message)
    

                # -------------------------------
                # 6) testing
                # -------------------------------
 #               if current_step % opt['train']['checkpoint_test'] == 0:
    
                    avg_psnr = 0.0
                    idx = 0
    
                    for test_data in test_loader:
                        idx += 1
                        image_name_ext = os.path.basename(test_data['L_path'][0])
                        img_name, ext = os.path.splitext(image_name_ext)
    
                        img_dir = os.path.join(opt['path']['images'], img_name)
                        util.mkdir(img_dir)
    
                        model.feed_data(test_data)
                        model.test()
    
                        visuals = model.current_visuals()
                        E_img = util.tensor2uint(visuals['E'])
                        H_img = util.tensor2uint(visuals['H'])
                        L_img = util.tensor2uint(visuals['L'])
                        
                        
                        val_psnr.update_psnr(calc_psnr(E_img, H_img), len(E_img))
                        val_ssim.update_ssim(calc_ssim(E_img, H_img), len(E_img))
                        logs = model.current_log()  # such as loss
                        
                        val_loss.update(logs['G_loss'], len(train_data['H']))
    
                        current_psnr = util.calculate_psnr(E_img, H_img, border=border)
    
                        avg_psnr += current_psnr
    
                    avg_psnr = avg_psnr / idx
  
                    val_loss.update_epoch(val_loss.avg)
                    val_psnr.update_epoch(val_psnr.avg)
                    val_ssim.update_epoch(val_ssim.avg)
    
                      
                    mlflow.log_metric("val loss", val_loss.per_epoch[epoch])
                    mlflow.log_metric("val psnr", val_psnr.per_epoch[epoch])
                    mlflow.log_metric("val ssim", val_ssim.per_epoch[epoch])   
                    
                    
                if current_step % opt['train']['checkpoint_save'] == 0:
                    logger.info('Saving the model.')
                    model.save(current_step)
        
    
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
