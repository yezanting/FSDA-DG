import argparse, os, sys, datetime, importlib

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='true'
import torch.optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from engine import train_warm_up,evaluate,train_one_epoch_SBF,train_one_epoch,prediction_wrapper
from losses import SetCriterion
import numpy as np
import random
from torch.optim import lr_scheduler
from dataloaders.TwoStreamBatchSampler import TwoStreamBatchSampler1
import math
from networks.net_factory import net_factory




def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')
    return seed

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )


    #revise the yaml file
    # list=["configs\efficientUnet_LEG_to_BSSFP.yaml"]



    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list()

    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=22,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    parser.add_argument('--labeled_bs', type=float, default=0.5, help='labeled_batch_size per gpu')
    parser.add_argument('--labelnum', type=float, default=0.5
                        , help='labeled data')
    parser.add_argument('--data', type=str, default="Abdiminal", help='data type')

    return parser
def patients_to_slices(dataset, patiens_num, data_number):
    ref_dict = None
    if "Adbominal" in dataset:
        label_size = data_number *  patiens_num
    elif "Cardiac":
        label_size = data_number * patiens_num
    else:
        print("Error")
    return int((label_size // 8) * 8)
def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))#获得target的值

class DataModuleFromConfig(torch.nn.Module):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers)

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args() #输出知道的参数和不知道的参数
    seed=seed_everything(opt.seed)
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
    if opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name=None
        raise ValueError('no config')

    nowname = now +f'_seed{seed}'+ name + opt.postfix
    logdir = os.path.join("logs", nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    visdir= os.path.join(logdir, "visuals")
    testvisdir= os.path.join(logdir, "testvisuals")
    for d in [logdir, cfgdir, ckptdir,visdir ]:
        os.makedirs(d, exist_ok=True)

#封装yaml
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    OmegaConf.save(config,os.path.join(cfgdir, "{}-project.yaml".format(now)))

    model_config = config.pop("model", OmegaConf.create())
    optimizer_config = config.pop('optimizer', OmegaConf.create())
    data_config = config.pop("data", OmegaConf.create())
    SBF_config = config.pop('saliency_balancing_fusion',OmegaConf.create())#是否启用saliency_balancing_fusion

    model = net_factory(net_type="mcnet2d_v1", in_chns=1, class_num=4)

    if torch.cuda.is_available():
        model=model.cuda()
#改掉了


    # if getattr(model_config.params, 'base_learning_rate') :
    #     bs, base_lr = config.data.params.batch_size, optimizer_config.base_learning_rate
    #     lr = bs * base_lr
    # else:

    bs, lr =data_config.params.batch_size, optimizer_config.learning_rate

    # # if getattr(model_config.params, 'pretrain') :
    # #     param_dicts = model.optim_parameters()
    # else:
    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1}]

    opt_params = {'lr': lr}
    for k in ['momentum', 'weight_decay']:
        if k in optimizer_config:
            opt_params[k] = optimizer_config[k]

    criterion = SetCriterion()

    print('optimization parameters: ', opt_params)
    opt2 = eval(optimizer_config['target'])(param_dicts, **opt_params)

    if optimizer_config.lr_scheduler =='lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + 0 - 50) / float(optimizer_config.max_epoch-50 + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(opt2, lr_lambda=lambda_rule)
    else:
        scheduler=None
        print('We follow the SSDG learning rate schedule by default, you can add your own schedule by yourself')
        raise NotImplementedError









#数据读入

    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup()
    print(len(data.datasets["train"]))#datasets 构建完成 返回的sample
    ###
    # #sample = {"images": img,
    #             "labels":lb[0].long(),
    #             "is_start": is_start,
    #             "is_end": is_end,
    #             "nframe": nframe,
    #             "scan_id": scan_id,
    #             "z_id": z_id,
    #             "aug_images": aug_img,
    #             }
    total_slices = len(data.datasets["train"])
    labeled_slice = patients_to_slices(opt.data, opt.labelnum, total_slices)#返回需要标签的样本数量
    print("Total silices is: {}, labeled slices is: {}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    labeled_each_batch = data_config.params.batch_size - opt.labeled_bs * data_config.params.batch_size
    batch_sampler = TwoStreamBatchSampler1(labeled_idxs, unlabeled_idxs, data_config.params.batch_size,
                                          data_config.params.batch_size - opt.labeled_bs * data_config.params.batch_size)
    def worker_init_fn(worker_id):
        random.seed(22 + worker_id)
    train_loader=DataLoader(data.datasets["train"], batch_sampler=batch_sampler,
                          num_workers=4,  persistent_workers=True,  pin_memory = True, worker_init_fn=worker_init_fn)

    val_loader=DataLoader(data.datasets["validation"], batch_size=data.batch_size,  num_workers=1)

    if data.datasets.get('test') is not None:
        test_loader=DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
        best_test_dice = 0
        test_phase=True
    else:
        test_phase=False
    from torch.nn.modules.loss import CrossEntropyLoss

    if getattr(optimizer_config, 'warmup_iter'):
        if optimizer_config.warmup_iter>0:
            train_warm_up(model, criterion, train_loader, opt, torch.device('cuda'), lr, optimizer_config.warmup_iter)
    cur_iter=0
    best_dice=0
    label_name=data.datasets["train"].all_label_names
#修改###
    assert optimizer_config.max_epoch > 0 or optimizer_config.max_iter > 0
    if optimizer_config.max_iter > 0:
        max_epoch= optimizer_config.max_iter // len(train_loader) + 1
        print('detect identified max iteration, set max_epoch to 999')
    else:
        max_epoch= optimizer_config.max_epoch
    iterator = tqdm(range(max_epoch), ncols=70)
########



    for cur_epoch in iterator:
        if SBF_config.usage:
            cur_iter = train_one_epoch_SBF(model, criterion,train_loader,opt2,torch.device('cuda'),cur_epoch,cur_iter, optimizer_config.max_iter, SBF_config, visdir, int(opt.labeled_bs *data_config.params.batch_size))
        else:
            cur_iter = train_one_epoch(model, criterion, train_loader, opt2, torch.device('cuda'), cur_epoch, cur_iter, optimizer_config.max_iter)
        if scheduler is not None:
            scheduler.step()

        # Save Bset model on val
        if (cur_epoch+1)%15==0:
            cur_dice = evaluate(model, val_loader, torch.device('cuda'))
            if np.mean(cur_dice)>best_dice:
                best_dice=np.mean(cur_dice)
                for f in os.listdir(ckptdir):
                    if 'val' in f:
                        os.remove(os.path.join(ckptdir,f))
                torch.save({'model': model.state_dict()}, os.path.join(ckptdir,f'val_best_epoch_{cur_epoch}.pth'))

            str=f'Epoch [{cur_epoch}]   '
            for i,d in enumerate(cur_dice):
                str+=f'Class {i}: {d}, '
            str+=f'Validation DICE {np.mean(cur_dice)}/{best_dice}'
            print(str)

        # Save latest model
        if (cur_epoch+1)%50==0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir,'latest{}.pth'.format(cur_epoch+1)))


        if cur_iter >= optimizer_config.max_iter and optimizer_config.max_iter>0:
            torch.save({'model': model.state_dict()}, os.path.join(ckptdir, 'latest.pth'))
            print(f'End training with iteration {cur_iter}')
            break

