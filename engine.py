import sys
from typing import Iterable
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tensorboardX import SummaryWriter

import nibabel as nib
import util.misc as utils
import functools
from tqdm import tqdm
import torch.nn.functional as F
from monai.metrics import compute_meandice
from torch.autograd import Variable
from dataloaders.saliency_balancing_fusion import get_SBF_map
from torch.nn.modules.loss import CrossEntropyLoss
from utils import losses, ramps
print = functools.partial(print, flush=True)
def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 200.0)

def train_warm_up(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, learning_rate:float, warmup_iteration: int = 1500):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    print_freq = 10
    cur_iteration=0
    while True:
        for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, 'WarmUp with max iteration: {}'.format(warmup_iteration))):
            for k,v in samples.items():
                if isinstance(samples[k],torch.Tensor):
                    samples[k]=v.to(device)
            cur_iteration+=1
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = cur_iteration/warmup_iteration*learning_rate * param_group["lr_scale"]

            img=samples['images']
            lbl=samples['labels']
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            if cur_iteration>=warmup_iteration:
                print(f'WarnUp End with Iteration {cur_iteration} and current lr is {optimizer.param_groups[0]["lr"]}.')
                return cur_iteration
        metric_logger.synchronize_between_processes()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1, grad_scaler=None):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        img = samples['images']
        lbl = samples['labels']

        if grad_scaler is None:
            pred = model(img)
            loss_dict = criterion.get_loss(pred,lbl)
            losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                pred = model(img)
                loss_dict = criterion.get_loss(pred,lbl)
                losses = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict.keys())
            optimizer.zero_grad()
            grad_scaler.scale(losses).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        metric_logger.update(**loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        cur_iteration+=1
        if cur_iteration>=max_iteration and max_iteration>0:
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return cur_iteration
def sharpening(P):
    T = 1/0.1
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def train_one_epoch_SBF(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, cur_iteration:int, max_iteration: int = -1,config=None,visdir=None,
                    labeled_bs: int = 8):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 25
    visual_freq = 100
    writer = SummaryWriter( 'F:/SLaug/SLAug-main/log')

    for i, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)

        GLA_img = samples['images']
        LLA_img = samples['aug_images']
        lbl = samples['labels']
        if cur_iteration % visual_freq == 0:
            visual_dict={}
            visual_dict2={}
            visual_dict['GLA']=GLA_img.detach().cpu().numpy()[0,0]
            visual_dict2['GLA']= np.expand_dims(visual_dict['GLA'], axis=0)
            visual_dict['LLA']=LLA_img.detach().cpu().numpy()[0,0]
            visual_dict2['LLA'] = np.expand_dims(visual_dict['LLA'], axis=0)
            visual_dict['GT']=lbl.detach().cpu().numpy()[0]
            visual_dict2['GT'] = np.expand_dims(visual_dict['GT'], axis=0)
            writer.add_image('train/GLA_Image', visual_dict2['GLA'], cur_iteration)
            writer.add_image('train/LLA_Image', visual_dict2['LLA'], cur_iteration)
            writer.add_image('train/GT', visual_dict2['GT'], cur_iteration)
        else:
            visual_dict=None

        input_var = Variable(GLA_img, requires_grad=True)

        optimizer.zero_grad()
        GLA_outputs = model(input_var)
        num_outputs = len(GLA_outputs)
        GLA_y_ori = torch.zeros((num_outputs,) + GLA_outputs[0].shape)
        GLA_y_pseudo_label = torch.zeros((num_outputs,) + GLA_outputs[0].shape)
        consistency_criterion = losses.mse_loss
        GLA_loss_seg = 0
        GLA_loss_seg_dice = 0
        loss_consist_GLA = 0
        loss_consist_SPF = 0
        for idx in range(num_outputs):
                GLA_y = GLA_outputs[idx][:labeled_bs, ...]
                # GLA_y_prob = F.softmax(GLA_y, dim=1)# norm in class-number layer
                GLA_loss_seg += criterion.get_ce(GLA_y, lbl[:labeled_bs][:].long()) # ce loss for labeled data in GLA_img
                GLA_loss_seg_dice += criterion.get_dc(GLA_y, lbl[:labeled_bs]) #dice loss for labeled data in GLA-img

                GLA_y_all = GLA_outputs[idx]
                GLA_y_prob_all = F.softmax(GLA_y_all, dim=1)
                GLA_y_ori[idx] = GLA_y_prob_all
                GLA_y_pseudo_label[idx] = sharpening(GLA_y_prob_all)
        for i in range(num_outputs):
            for j in range(num_outputs):
                if i != j:
                    loss_consist_GLA += consistency_criterion(GLA_y_ori[i], GLA_y_pseudo_label[j])
        consistency_weight = get_current_consistency_weight(cur_iteration // 150)
        GLA_loss =  GLA_loss_seg_dice + consistency_weight * loss_consist_GLA
        optimizer.zero_grad()
        GLA_loss.backward()
        # saliency
        gradient = torch.sqrt(torch.mean(input_var.grad ** 2, dim=1, keepdim=True)).detach()

        saliency = get_SBF_map(gradient, config.grid_size)
        if visual_dict is not None:
            visual_dict['GLA_pred'] = torch.argmax(GLA_outputs[1], 1).cpu().numpy()[0]
            visual_dict2['GLA_pred'] = np.expand_dims(visual_dict['GLA_pred'], axis=0)
            writer.add_image('train/GLA_pred', visual_dict2['GLA_pred'], cur_iteration)
        if visual_dict is not None:
            visual_dict['GLA_saliency'] = saliency.detach().cpu().numpy()[0, 0]
            visual_dict2['GLA_saliency'] = np.expand_dims(visual_dict['GLA_saliency'], axis=0)
            writer.add_image('train/GLA_saliency', visual_dict2['GLA_saliency'], cur_iteration)
        mixed_img = GLA_img.detach() * saliency + LLA_img * (1 - saliency)
        if visual_dict is not None:
            visual_dict['SBF'] = mixed_img.detach().cpu().numpy()[0, 0]
            visual_dict2['SBF'] = np.expand_dims(visual_dict['SBF'], axis=0)
            writer.add_image('train/SBF', visual_dict2['SBF'], cur_iteration)
        aug_var = Variable(mixed_img, requires_grad=True)
        aug_outputs = model(aug_var)
        aug_loss_dict = criterion.get_loss(torch.tensor(aug_outputs[0]), lbl)
        aug_num_outputs = len(aug_outputs)
        aug_y_ori = torch.zeros((aug_num_outputs,) + aug_outputs[0].shape)
        aug_y_pseudo_label = torch.zeros((aug_num_outputs,) + aug_outputs[0].shape)
        consistency_criterion = losses.mse_loss
        aug_loss_seg = 0
        aug_loss_seg_dice=0
        for idx in range(aug_num_outputs):
            aug_y = aug_outputs[idx][:labeled_bs, ...]
            # aug_y_prob = F.softmax(aug_y, dim=1)
            aug_loss_seg += criterion.get_ce(aug_y, lbl[:labeled_bs][:].long())  # ce loss for labeled data in GLA_img
            aug_loss_seg_dice += criterion.get_dc(aug_y,
                                                  lbl[:labeled_bs].unsqueeze(1))  # dice loss for labeled data in GLA-img

            aug_y_all = aug_outputs[idx]
            aug_y_prob_all = F.softmax(aug_y_all, dim=1)
            aug_y_ori[idx] = aug_y_prob_all
            aug_y_pseudo_label[idx] = sharpening(aug_y_prob_all)
        loss_consist_aug = 0
        for i in range(num_outputs):
            for j in range(num_outputs):
                if i != j:
                    loss_consist_aug += consistency_criterion(aug_y_ori[i], aug_y_pseudo_label[j])
        consistency_weight = get_current_consistency_weight(cur_iteration // 150)
        loss_consist_aug_gla = 0
        for i in range(num_outputs):
            loss_consist_aug_gla += consistency_criterion(GLA_y_pseudo_label[i].data, aug_y_pseudo_label[i])

        aug_loss = aug_loss_seg_dice + consistency_weight * (loss_consist_aug+loss_consist_aug_gla)

        aug_loss.backward()
        optimizer.step()

        loss_dict = criterion.get_loss(aug_outputs[0], lbl)
        if visual_dict is not None:
            visual_dict['SBF_pred'] = torch.argmax(aug_outputs[0], 1).cpu().numpy()[0]
        all_loss_dict={}
        for k in loss_dict.keys():
            if k not in criterion.weight_dict:continue
            all_loss_dict[k]=loss_dict[k]
            all_loss_dict[k+'_aug']=aug_loss_dict[k]

            metric_logger.update(**all_loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])


        if cur_iteration>=max_iteration and max_iteration>0:
            break

        if visdir is not None and cur_iteration%visual_freq==0:
            fs=int(len(visual_dict)**0.5)+1
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs,fs,idx+1)
                plt.title(k)
                plt.axis('off')
                if k not in ['GT','GLA_pred','SBF_pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{visdir}/{cur_iteration}.png')
            plt.close()
        cur_iteration+=1


    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    writer.close()
    return cur_iteration


@torch.no_grad()
def evaluate(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    def convert_to_one_hot(tensor,num_c):
        return F.one_hot(tensor,num_c).permute((0,3,1,2))
    dices=[]
    for samples in data_loader:
        for k, v in samples.items():
            if isinstance(samples[k], torch.Tensor):
                samples[k] = v.to(device)
        img = samples['images']
        lbl = samples['labels']
        logits = model(img)[0]
        num_classes=logits.size(1)
        pred=torch.argmax(logits,dim=1)
        one_hot_pred=convert_to_one_hot(pred,num_classes)
        one_hot_gt=convert_to_one_hot(lbl,num_classes)
        dice=compute_meandice(one_hot_pred,one_hot_gt,include_background=False)
        dices.append(dice.cpu().numpy())
    dices=np.concatenate(dices,0)
    dices=np.nanmean(dices,0)
    return dices

def   prediction_wrapper(model, test_loader, epoch, label_name, mode = 'base', test_visual_dir= '' , save_prediction = False ):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    model.eval()
    sys.path.append(os.getcwd())
    test_visdir = test_visual_dir
    z=0
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        # recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            z = z+1
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['images'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  )).cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  )).cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['labels'].shape[0] == 1 # enforce a batchsize of 1

            img = batch['images'].cuda()
            gth = batch['labels'].cuda()
#chose the output-d
            pred = model(img)[0]
            pred=torch.argmax(pred,1)#每一个单独预测，然后整合
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_img[:,:,slice_idx] = batch['images'][0, 0,...].numpy()
            slice_idx += 1

#visual
            visual_dict={}

            # visual_dict['img']=curr_img.transpose(2,0,1)
            visual_dict['img'] = img[0,0].detach().cpu().numpy()
            print_img = (visual_dict['img']-np.min(visual_dict['img']))/(np.max(visual_dict['img'])-np.min(visual_dict['img']))
            print_img = Image.fromarray(np.uint8(print_img*255))
            if not os.path.exists(f'{test_visdir}'):
                os.makedirs(os.path.dirname(f'{test_visdir}/latest.pth'))
            print_img.save(f'{test_visdir}/{z}_img.png')
            visual_dict['gt']=gth[0].detach().cpu().numpy()
            print_gt = (visual_dict['gt']-np.min(visual_dict['gt']))/(np.max(visual_dict['gt'])-np.min(visual_dict['img']))
            print_gt = Image.fromarray(np.uint8(print_gt*255))
            print_gt.save(f'{test_visdir}/{z}_gt.png')
            visual_dict['pred'] = pred[0].detach().cpu().numpy()
            print_pred = (visual_dict['pred']-np.min(visual_dict['pred']))/(np.max(visual_dict['pred'])-np.min(visual_dict['pred']))
            print_pred = Image.fromarray(np.uint8(print_pred*255))
            print_pred.save(f'{test_visdir}/{z}_pred.png')
            fs = int(len(visual_dict) ** 0.5) + 1
            # nii_image2 = nib.load(r'F:\SLaug\SLAug-main\data\abdominal\CHAOST2\processed\image_1.nii.gz')
            #
            # affine = nii_image2.affine
            for idx, k in enumerate(visual_dict.keys()):
                plt.subplot(fs, fs, idx + 1)
                plt.title(k)
                plt.axis('off')
                if k not in ['img', 'gt', 'pred']:
                    plt.imshow(visual_dict[k], cmap='gray')
                else:
                    plt.imshow(visual_dict[k], vmin=0, vmax=4)
            plt.tight_layout()
            plt.savefig(f'{test_visdir}/{z}.png')
            plt.close()
                # if k in ['img']:
                #     ni_img = nib.Nifti1Image(visual_dict[k], affine)
                #     if not os.path.exists(f'{test_visdir}/{z}_img.nii'):
                #         # os.makedirs(os.path.dirname(f'{test_visdir}/{z}_img.nii'))
                #         nib.save(ni_img, f'{test_visdir}/{z}_img.nii')
                #     else:
                #         nib.save(ni_img, f'{test_visdir}/{z}_img.nii')
                #
                # if k in ['gt']:
                #     ni_img = nib.Nifti1Image(visual_dict[k], affine)
                #     # if not os.path.exists(f'{test_visdir}/{z}_gt.nii'):
                #     #     os.makedirs(os.path.dirname(f'{test_visdir}/{z}_gt.nii'))
                #     nib.save(ni_img, f'{test_visdir}/{z}_gt.nii')
                #
                # if k in ['pred']:
                #     ni_img = nib.Nifti1Image(visual_dict[k], affine)
                #     # if not os.path.exists(f'{test_visdir}/{z}_pred.nii'):
                #     #     os.makedirs(os.path.dirname(f'{test_visdir}/{z}_pred.nii'))
                #     nib.save(ni_img, f'{test_visdir}/{z}_pred.nii')



            # nii_image2 = nib.load(r'F:\SLaug\SLAug-main\data\abdominal\CHAOST2\processed\image_1.nii.gz')
            #
            # affine  = nii_image2.affine
            # for idx, k in enumerate(visual_dict.keys()):
            #
            #     if k in ['img']:
            #         ni_img = nib.Nifti1Image(visual_dict[k],affine )
            #         if not os.path.exists(f'{test_visdir}/{z}_img.nii'):
            #             # os.makedirs(os.path.dirname(f'{test_visdir}/{z}_img.nii'))
            #             nib.save(ni_img, f'{test_visdir}/{z}_img.nii')
            #         else:
            #             nib.save(ni_img, f'{test_visdir}/{z}_img.nii')
            #
            #     if k in ['gt']:
            #         ni_img = nib.Nifti1Image(visual_dict[k],affine )
            #         # if not os.path.exists(f'{test_visdir}/{z}_gt.nii'):
            #         #     os.makedirs(os.path.dirname(f'{test_visdir}/{z}_gt.nii'))
            #         nib.save(ni_img, f'{test_visdir}/{z}_gt.nii')
            #
            #     if k in ['pred']:
            #         ni_img = nib.Nifti1Image(visual_dict[k],affine )
            #         # if not os.path.exists(f'{test_visdir}/{z}_pred.nii'):
            #         #     os.makedirs(os.path.dirname(f'{test_visdir}/{z}_pred.nii'))
            #         nib.save(ni_img, f'{test_visdir}/{z}_pred.nii')



                # plt.subplot(fs, fs, idx + 1)
                # plt.title(k)
                # plt.axis('off')

            # plt.tight_layout()
            # plt.savefig(f'{test_visdir}/{idx}.png')
            # plt.close()

###


            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                # if opt.phase == 'test':
                #     recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name),label_name)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, label_name):
    """
    Evaluatation and arrange predictions
    """
    def convert_to_one_hot2(tensor,num_c):
        return F.one_hot(tensor.long(),num_c).permute((3,0,1,2)).unsqueeze(0)

    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures
    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices=compute_meandice(y_pred=convert_to_one_hot2(pred_,nclass),y=convert_to_one_hot2(gth_,nclass),include_background=True).cpu().numpy()[0].tolist()

        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc
    print("Overall std dice by sample {:06.5f}".format(dsc_table[:, 1:].std()))
    print("Overall mean dice by sample {:06.5f}".format( dsc_table[:,1:].mean())) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)
    print('per domain resutls:', overall_by_domain)
    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    return error_dict, dsc_table, domain_names

