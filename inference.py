import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import SimpleITK as sitk
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nmse
from fusion import fusion_image, Himage1_Limage2, Limage1_Himage2
# from pytorch_ssim import ssim as ssim
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim
import time




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_16_128_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    # num_params = sum(param.numel() for param in diffusion.parameters())
    # print(num_params)

    logger.info('Initial Model Finished')
    # logger.info('num_params',num_params)


    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0
    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    cnt, cnt3d = 0, 0
    EPETimg = np.zeros([128, 128, 128])
    SPETimg = np.zeros([128, 128, 128])
    IPETimg = np.zeros([128, 128, 128])
    FPETimg = np.zeros([128, 128, 128])
    DPETimg = np.zeros([128, 128, 128])
    CC_GT_PETimg = np.zeros([128, 128, 128])
    CC_PRED_PETimg = np.zeros([128, 128, 128])
    CC_Stage2 = np.zeros([128, 128, 128])


    num = 0
    total_d_psnr, total_d_ssim, total_d_nmse = [], [], []
    total_ip_psnr,total_ip_ssim,total_ip_nmse = [],[],[]
    total_clean2_psnr,total_clean2_ssim,total_clean2_nmse = [],[],[]
    total_final_psnr,total_final_ssim,total_final_nmse = [],[],[]

    time_start = time.time()
    total_time=[]
    for _, val_data in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        time_start = time.time()
        diffusion.test(continous=False)
        time_end = time.time()
        visuals = diffusion.get_current_visuals(need_LR=False)
        total_time.append(time_end-time_start)
        print('time cost', time_end - time_start, 's')
        image_s = np.squeeze(visuals['HR'].cpu().detach().numpy())
        clean2 = np.squeeze(visuals['clean2'].cpu().detach().numpy())#dm预测
        finalclean = np.squeeze(visuals['finalclean'].cpu().detach().numpy())
        IP = np.squeeze(visuals['IP'].cpu().detach().numpy())#粗预测
        dirty=np.squeeze(visuals['INF'].cpu().detach().numpy())#原始脏图片
        cc_gt=np.squeeze(visuals['cc_gt'].cpu().detach().numpy())#cc的gt
        cc_pred=np.squeeze(visuals['cc_pred'].cpu().detach().numpy())#cc的预测
        cc_stage2 = np.squeeze(visuals['cc_stage2'].cpu().detach().numpy())
        EPETimg[cnt, :, :] = clean2
        IPETimg[cnt, :, :] = IP
        FPETimg[cnt, :, :] = finalclean
        SPETimg[cnt, :, :] = image_s#HR
        DPETimg[cnt, :, :] = dirty#dirty
        CC_GT_PETimg[cnt, :, :] = cc_gt#cc的gt
        CC_PRED_PETimg[cnt, :, :] = cc_pred#cc的预测
        CC_Stage2[cnt, :, :] = cc_stage2


        cnt += 1

        if cnt == 128:
    
            cnt = 0
            cnt3d += 1
            chann, weight, height = EPETimg.shape
            # FPETimg=fusion_image(EPETimg,IPETimg)
            FPETimg = Himage1_Limage2(EPETimg, IPETimg)
            # FPETimg = Limage1_Himage2(EPETimg, IPETimg)
            print(FPETimg.shape)


            for c in range(chann):  # 遍历高
                for w in range(weight):  # 遍历宽
                    for h in range(height):
                        if EPETimg[c][w][h] <= 0.05:
                            EPETimg[c][w][h] = 0
                        if IPETimg[c][w][h] <= 0.05:
                            IPETimg[c][w][h] = 0
                        if FPETimg[c][w][h] <= 0.05:
                            FPETimg[c][w][h] = 0
                            SPETimg[c][w][h] = 0


            y = np.nonzero(SPETimg)  # 取非黑色部分
            SPETimg_1 = SPETimg[y]#gt
            EPETimg_1 = EPETimg[y]#clean2
            IPETimg_1 = IPETimg[y]#粗迭代
            FPETimg_1 = FPETimg[y]#fincal
            DPETimg_1 = DPETimg[y]

            print(IPETimg_1.shape)

            d_psnr = psnr(DPETimg_1, SPETimg_1, data_range=np.max([DPETimg_1.max(), SPETimg_1.max()]) - np.min(
                [DPETimg_1.min(), SPETimg_1.min()]))
            # ip_psnr = psnr(IPETimg_1, SPETimg_1, data_range=1)
            d_ssim = ssim(DPETimg, SPETimg)
            d_nmse = nmse(DPETimg, SPETimg)

            ip_psnr = psnr(IPETimg_1, SPETimg_1, data_range=np.max([IPETimg_1.max(), SPETimg_1.max()]) - np.min([IPETimg_1.min(), SPETimg_1.min()]))
            # ip_psnr = psnr(IPETimg_1, SPETimg_1, data_range=1)
            ip_ssim = ssim(IPETimg, SPETimg)
            ip_nmse = nmse(IPETimg, SPETimg)**2

            clean2_psnr = psnr(EPETimg_1, SPETimg_1, data_range=np.max([EPETimg_1.max(), SPETimg_1.max()]) - np.min([EPETimg_1.min(), SPETimg_1.min()]))
            # clean2_psnr = psnr(EPETimg_1, SPETimg_1, data_range= 1)
            clean2_ssim = ssim(EPETimg, SPETimg)
            clean2_nmse = nmse(EPETimg, SPETimg)**2

            final_psnr = psnr(FPETimg_1, SPETimg_1, data_range=np.max([FPETimg_1.max(), SPETimg_1.max()]) - np.min([FPETimg_1.min(), SPETimg_1.min()]))
            # final_psnr = psnr(FPETimg_1, SPETimg_1, data_range= 1)
            final_ssim = ssim(FPETimg, SPETimg)
            final_nmse = nmse(FPETimg, SPETimg)**2

            print("------",EPETimg.max(),SPETimg.max())

            print('D_PSNR: {:6f} D_SSIM: {:6f} D_NMSE: {:6f} '.format(d_psnr, d_ssim, d_nmse))
            print('IP_PSNR: {:6f} IP_SSIM: {:6f} IP_NMSE: {:6f} '.format(ip_psnr,ip_ssim,ip_nmse))
            print('clean2_PSNR: {:6f} clean2_SSIM: {:6f}  clean2_NMSE: {:6f} '.format(clean2_psnr, clean2_ssim, clean2_nmse))
            print('final_PSNR: {:6f} final_SSIM: {:6f} final_NMSE: {:6f} '.format(final_psnr,final_ssim,final_nmse))

            # plt.imshow(FPETimg, cmap='gray')
            # plt.show()
            # if final_ssim > 0.94 :
            num += 1

            total_d_psnr.append(d_psnr)
            total_d_ssim.append(d_ssim)
            total_d_nmse.append(d_nmse)

            total_ip_psnr.append(ip_psnr)
            total_ip_ssim.append(ip_ssim)
            total_ip_nmse.append(ip_nmse)

            total_clean2_psnr.append(clean2_psnr)
            total_clean2_ssim.append(clean2_ssim)
            total_clean2_nmse.append(clean2_nmse)


            total_final_psnr.append(final_psnr)
            total_final_ssim.append(final_ssim)
            total_final_nmse.append(final_nmse)

            Metrics.save_img(EPETimg, '{}/{}_{}_clean2.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(FPETimg, '{}/{}_{}_final.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(IPETimg,'{}/{}_{}_IP.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(DPETimg,'{}/{}_{}_dirty.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(SPETimg, '{}/{}_{}_hr.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(CC_GT_PETimg, '{}/{}_{}_cc_gt.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(CC_PRED_PETimg, '{}/{}_{}_cc_pred.img'.format(result_path, current_step, cnt3d))
            Metrics.save_img(CC_Stage2, '{}/{}_{}_cc_stage2.img'.format(result_path, current_step, cnt3d))

            avg_d_psnr = np.mean(total_d_psnr)
            avg_d_ssim = np.mean(total_d_ssim)
            avg_d_nmse = np.mean(total_d_nmse)

            avg_ip_psnr=np.mean(total_ip_psnr)
            avg_ip_ssim=np.mean(total_ip_ssim)
            avg_ip_nmse=np.mean(total_ip_nmse)

            avg_clean2_psnr=np.mean(total_clean2_psnr)
            avg_clean2_ssim=np.mean(total_clean2_ssim)
            avg_clean2_nmse=np.mean(total_clean2_nmse)

            avg_final_psnr=np.mean(total_final_psnr)
            avg_final_ssim=np.mean(total_final_ssim)
            avg_final_nmse=np.mean(total_final_nmse)

            print(': Avg. D_PSNR: {:6f} D_SSIM: {:6f} D_NMSE: {:6f} '.format(avg_d_psnr, avg_d_ssim, avg_d_nmse))
            print(': Avg. IP_PSNR: {:6f} IP_SSIM: {:6f} IP_NMSE: {:6f} '.format(avg_ip_psnr,avg_ip_ssim,avg_ip_nmse))
            print(': Avg. clean2_PSNR: {:6f} clean2_SSIM: {:6f} clean2_NMSE: {:6f} '.format(avg_clean2_psnr, avg_clean2_ssim, avg_clean2_nmse))
            print(': Avg. final_PSNR: {:6f} final_SSIM: {:6f} final_NMSE: {:6f}'.format(avg_final_psnr, avg_final_ssim, avg_final_nmse))
            print('AVGtime:',np.mean(total_time))
            print('num:', num)
            if wandb_logger and opt['log_infer']:
                wandb_logger.log_eval_table(commit=True)
