import logging
from collections import OrderedDict

# import time
from itertools import chain
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
logger = logging.getLogger('base')
from thop import profile

class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        # self.netP = self.set_device(networks.define_P(opt))#粗预测的unet
        self.netG = self.set_device(networks.define_G(opt))##需要diffusion的unet
        logger.info('Initialization method Y')

        self.netY = self.set_device(networks.define_Y(opt))#替换的adfnet
        logger.info('Initialization method Y')
        self.schedule_phase = None
        self.lr = opt['train']["optimizer"]["lr"]
        # set loss and load resume state
        self.loss_func = nn.L1Loss(reduction='sum').to(self.device)
        self.set_loss()
        self.set_new_noise_schedule(
            opt['model']['beta_schedule']['train'], schedule_phase='train')
        if self.opt['phase'] == 'train':
            self.netG.train()
            self.netY.train()

          
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                optim_params_P = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
                # for k, v in self.netP.named_parameters():
                    # v.requires_grad = False
                    # if k.find('transformer') >= 0:
                    #     v.requires_grad = True
                    #     v.data.zero_()
                    #     optim_params.append(v)
                    #     logger.info(
                    #         'Params [{:s}] initialized to 0 and will optimize.'.format(k))
                for k, v in self.netY.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            'Params [{:s}] initialized to 0 and will optimize.'.format(k))
               
            else:
                optim_params = list(self.netG.parameters())
                # optim_params_P = list(self.netP.parameters())
                optim_params_Y = list(self.netY.parameters())
                
            self.optG = torch.optim.Adam(
                optim_params, lr=opt['train']["optimizer"]["lr"] , weight_decay=0.0001)
            # self.optP = torch.optim.Adam(
            #     optim_params_P, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
            self.optY = torch.optim.Adam(
                optim_params_Y, lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
            self.optCOMBINED = torch.optim.Adam(params=chain(optim_params_Y,optim_params),lr=opt['train']["optimizer"]["lr"], weight_decay=0.0001)
            self.log_dict = OrderedDict()
        self.load_network()

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters1(self):
        self.optG.zero_grad()
        self.optY.zero_grad()

        # start_time=time.time()
        self.initial_predict()#粗预测
        # finish_time=time.time()
        # print(f"粗预测耗时:{start_time-finish_time} 秒")
        # calculate residual as x_start
        self.data['IP'] = self.IP#粗迭代
        self.data['noise1'] = self.data['SR']-self.data['IP']#noise1:dirty-clean估计（output1）
        self.data['noise_gt']= self.data['SR']-self.data['HR']#noise的gt:dirty-clean

        # start_time=time.time()
        l_pix = self.netG(self.data)
        # finish_time=time.time()
        # print(f"dm耗时:{start_time-finish_time} 秒")
        # print('l_pix',l_pix)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = (l_pix[0].sum()+l_pix[1].sum())/int(b*c*h*w)+loss_guide
        # iploss=(l_pix[0].sum())/int(b*c*h*w)
        # dmloss=(l_pix[1].sum())/int(b*c*h*w)
        iploss=l_pix[0].sum()
        dmloss=l_pix[1].sum()/int(h)
        unetloss=l_pix[2].sum()
        l_pix = iploss+dmloss+unetloss

        iploss.backward(retain_graph=True)
        self.optY.step()

        unetloss.backward(retain_graph=True)
        self.optG.step()

        dmloss.backward()
        self.optY.step()
        self.optG.step()
        # set log
        self.log_dict['l_total'] = l_pix.item()
        self.log_dict['IP_loss'] = iploss.item()
        self.log_dict['DM_loss'] = dmloss.item()
        self.log_dict['unet_loss'] = unetloss.item()


    def optimize_parameters4(self):
        self.optG.zero_grad()
        self.optY.zero_grad()

        # start_time=time.time()
        self.initial_predict()#粗预测
        # finish_time=time.time()
        # print(f"粗预测耗时:{start_time-finish_time} 秒")
        # calculate residual as x_start
        self.data['IP'] = self.IP#粗迭代
        self.data['noise1'] = self.data['SR']-self.data['IP']#noise1:dirty-clean估计（output1）
        self.data['noise_gt']= self.data['SR']-self.data['HR']#noise的gt:dirty-clean

        # start_time=time.time()
        l_pix = self.netG(self.data)
        # finish_time=time.time()
        # print(f"dm耗时:{start_time-finish_time} 秒")
        # print('l_pix',l_pix)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = (l_pix[0].sum()+l_pix[1].sum())/int(b*c*h*w)+loss_guide
        # iploss=(l_pix[0].sum())/int(b*c*h*w)
        # dmloss=(l_pix[1].sum())/int(b*c*h*w)
        iploss=l_pix[0].sum()
        dmloss=l_pix[1].sum()
        l_pix = iploss+dmloss

        iploss.backward(retain_graph=True)
        self.optY.step()

        dmloss.backward()
        self.optG.step()
        # self.optY.step()

        # set log
        self.log_dict['l_total'] = l_pix.item()
        self.log_dict['IP_loss'] = iploss.item()
        self.log_dict['DM_loss'] = dmloss.item()

    def optimize_parameters(self):
        self.initial_predict()#粗预测
        lossy=self.loss_func(self.data['HR'],self.IP)
        self.optY.zero_grad()
        lossy.backward(retain_graph=True)
        self.optY.step()
        # finish_time=time.time()
        # print(f"粗预测耗时:{start_time-finish_time} 秒")
        # calculate residual as x_start
        self.data['IP'] = self.IP.detach()#粗迭代
        self.data['cc1'] = self.data['IP']-self.data['SR']#noise1:dirty-clean估计（output1）
        self.data['cc_gt']= self.data['HR']-self.data['SR']#noise的gt:dirty-clean

        # # 计算params和FLOPs
        # flops, params = profile(self.netG, (self.data,))
        # print('netG flops: ', flops, 'params: ', params)

        # start_time=time.time()
        l_pix = self.netG(self.data)
        # finish_time=time.time()
        # print(f"dm耗时:{start_time-finish_time} 秒")
        # print('l_pix',l_pix)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = (l_pix[0].sum()+l_pix[1].sum())/int(b*c*h*w)+loss_guide
        # iploss=(l_pix[0].sum())/int(b*c*h*w)
        # dmloss=(l_pix[1].sum())/int(b*c*h*w)
        iploss=l_pix[0].sum()
        dmloss=l_pix[1].sum()
        l_pix = iploss+dmloss


        self.optG.zero_grad()
        dmloss.backward()
        self.optG.step()
        # self.optY.step()

        # set log
        self.log_dict['l_total'] = l_pix.item()
        self.log_dict['IP_loss'] = iploss.item()
        self.log_dict['DM_loss'] = dmloss.item()


    def optimize_parameters2(self):
        self.optG.zero_grad()
        self.optY.zero_grad()
        # self.optCOMBINED.zero_grad()

        # start_time=time.time()
        self.initial_predict()#粗预测
        # finish_time=time.time()
        # print(f"粗预测耗时:{start_time-finish_time} 秒")
        # calculate residual as x_start
        self.data['IP'] = self.IP#粗迭代
        self.data['noise1'] = self.data['SR']-self.data['IP']#noise1:dirty-clean估计（output1）
        self.data['noise_gt']= self.data['SR']-self.data['HR']#noise的gt:dirty-clean

        # start_time=time.time()
        l_pix = self.netG(self.data)
        # finish_time=time.time()
        # print(f"dm耗时:{start_time-finish_time} 秒")
        # print('l_pix',l_pix)
        # need to average in multi-gpu
        b, c, h, w = self.data['HR'].shape
        # l_pix = (l_pix[0].sum()+l_pix[1].sum())/int(b*c*h*w)+loss_guide
        iploss=(l_pix[0].sum())/int(b*c*h)
        dmloss=(l_pix[1].sum())/int(b*c*h)
        # iploss=l_pix[0].sum()
        # dmloss=l_pix[1].sum()
        l_pix = iploss+dmloss

        l_pix.backward()
        # update all networks
        self.optG.step()
        self.optY.step()
        # set log
        self.log_dict['l_total'] = l_pix.item()
        self.log_dict['IP_loss'] = iploss.item()
        self.log_dict['DM_loss'] = dmloss.item()

    def initial_predict(self):
        # 计算params和FLOPs
        flops, params = profile(self.netY, (self.data['SR'],))
        print('netY flops: ', flops, 'params: ', params)

        self.IP = self.netY(self.data['SR'])





    def test(self, continous=False):
        self.netG.eval()
        self.netY.eval()

        with torch.no_grad():
            self.IP = self.netY(self.data['SR'])  # stage1的output
            self.data['cc1'] = self.IP-self.data['SR']

            if isinstance(self.netG, nn.DataParallel):
                self.pred_cc = self.netG.module.super_resolution(
                    self.data['cc1'],continous)
            else:
                self.pred_cc = self.netG.super_resolution(
                    self.data['cc1'], continous)
            self.clean2=self.data['SR']+self.pred_cc
            # print(self.data['SR'].max(),self.data['noise1'].max(),self.IP.max(),self.pred_noise.max(),self.clean2.max())

            self.finalclean=0.7*self.IP+0.3*self.clean2
        self.netG.train()
        self.netY.train()
       

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)


    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['SAM'] = self.SR.detach().float().cpu()
        else:
            out_dict['IP'] = self.IP.detach().float().cpu()
            out_dict['cc_gt'] = self.data['HR'].detach().float().cpu()-self.data['SR'].detach().float().cpu()
            out_dict['cc_pred'] = self.IP.detach().float().cpu()-self.data['SR'].detach().float().cpu()
            out_dict['clean2'] = self.clean2.detach().float().cpu()
            out_dict['cc_stage2'] = self.clean2.detach().float().cpu()-self.data['SR'].detach().float().cpu()
            out_dict['finalclean'] = self.finalclean.detach().float().cpu()
            out_dict['INF'] = self.data['SR'].detach().float().cpu()
            out_dict['HR'] = self.data['HR'].detach().float().cpu()
            if need_LR and 'LR' in self.data:
                out_dict['LR'] = self.data['LR'].detach().float().cpu()
            else:
                out_dict['LR'] = out_dict['INF']
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_state_dict(self, net, iter_step, epoch, name):
        if isinstance(net, nn.DataParallel):
            net = net.module
        state_dict = net.state_dict()

        # 计算并打印可训练模型参数的数量
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info(f"{name}可训练模型参数量: {trainable_params}")
        # 计算并打印模型的参数量
        total_params = sum(p.numel() for p in net.parameters())
        logger.info(f"{name}模型总参数量: {total_params}")
        # 计算并打印可训练模型参数的数量（MParam）
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info(f"{name}可训练模型参数量（MParam）: {trainable_params / 1e6:.2f}")


        for key, param in state_dict.items():
            state_dict[key] = param.cpu()

        gen_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_gen.pth')
        torch.save(state_dict, gen_path)
        return gen_path

    def save_optimizer_state(self, opt_net, iter_step, epoch, name):
        opt_state = {'epoch': epoch, 'iter': iter_step, 'scheduler': None, 'optimizer': opt_net.state_dict()}
        opt_path = os.path.join(self.opt['path']['checkpoint'], f'I{iter_step}_E{epoch}_{name}_opt.pth')
        torch.save(opt_state, opt_path)


    def save_network(self, epoch, iter_step):
        networks = [
            # (self.netP, self.optP, "PreNet"),
            (self.netY, self.optY, "PreNet"),
            (self.netG, self.optG, "DenoiseNet"),
           
        ]

        for net, opt_net, name in networks:
            gen_path = self.save_state_dict(net, iter_step, epoch, name)
            self.save_optimizer_state(opt_net, iter_step, epoch, name)
      


        logger.info(f'Saved model in [{gen_path}] ...')

    def load_network_state(self, network, load_path, model_name):
        gen_path = f'{load_path}_{model_name}_gen.pth'
        logger.info(f'Loading pretrained model for {model_name} [{gen_path}] ...')
        if isinstance(network, nn.DataParallel):
            network = network.module
        network.load_state_dict(torch.load(gen_path), strict=(not self.opt['model']['finetune_norm']))
        return network

    def load_optimizer_state(self, opt_net, load_path, model_name):
        opt_path = f'{load_path}_{model_name}_opt.pth'
        opt = torch.load(opt_path)
        opt_net.load_state_dict(opt['optimizer'])
        self.begin_step = opt['iter']
        self.begin_epoch = opt['epoch']

    def load_network(self):
        if self.opt['path']['resume_state'] is not None:
            load_path = self.opt['path']['resume_state']
            if self.opt['phase'] == 'train':
                networks = [
                # (self.netP, self.optP, "PreNet"),
                (self.netY, self.optY, "PreNet"),

                (self.netG, self.optG, "DenoiseNet"),
             
                ]

                for net, opt_net, name in networks:
                    self.load_optimizer_state(opt_net, load_path, name)
            else:
                networks = [
                    # (self.netP,  "PreNet"),
                    (self.netY,  "PreNet"),

                    (self.netG, "DenoiseNet"),
                  
                ]
                for net, name in networks:
                    net = self.load_network_state(net, load_path, name)
        

                
