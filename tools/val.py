# Description: This script is used to evaluate the performance of the model on the validation dataset.
# Modified from val.py in original YOLOv5, by zhang-xd18

import argparse
import os
import sys
import yaml
import torch
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import math
from tqdm import tqdm

from dataset import create_Env_dataloader
from utils import logger
from models import create_model
from utils.general import Profile, clustering, print_args, increment_path
from utils.init import init_device
from utils.metrics import AverageMeter, para2pos, cal_metric_pos

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format

def run(dataloader=None,    # validation data loader
        save_dir=Path(''),  # save results to save_dir
        model=None, # validation model
        device='',  # cuda device, i.e. 0 or cpu
        conf_thres=5e-1,  # confidence threshold
        batch_size=64,  # batch size
        imgsz=64,  # image size
        epoch_id=-1,    # id for epoch
        detect_thres=1):

    # Device
    device = next(model.parameters()).device  # get model device, PyTorch model
    cuda = device.type != 'cpu' # device config
    # Directories
    if logger._log_file == None:
        logger.set_file(save_dir / "log.txt")
    # Model
    model.eval()
    # Dataloader
    assert dataloader is not None, 'Dataset is not configured'
    # Running
    seen = 0    # number of processed images
    dt = Profile(), Profile(), Profile()  # time profiles config
    pbar = tqdm(dataloader, bar_format=TQDM_BAR_FORMAT)  # progress bar config
    logger.info(('\n' + '%15s' * 5) % ('Epoch', 'F1-score', 'Pd', 'RMSE', 'Size'))
    # Results
    iter_f1score = AverageMeter("f1-score")
    iter_pd = AverageMeter("pd")
    iter_rmse = AverageMeter("rmse")

    # Validation
    for batch_i, (im, targets, pos, nl) in enumerate(pbar):
        # preprocess
        with dt[0]:
            if cuda:
                im = im.to(device, non_blocking=True)
                pos = pos.to(device)
                targets = targets[targets[:,1] > 0, :]
                assert targets.shape[0] == nl.sum(), f"number of targets do not match"
                targets = targets.to(device)
            _, nc, height, width = im.shape 

        with dt[1]:
            preds = model(im) 
            
        # merge
        targets[:, 1:] *= torch.tensor((height, width), device=device)
        with dt[2]:
            preds = clustering(prediction=preds, conf_thres=conf_thres)  # merge
            
        # calculate metrics
        for si, pred in enumerate(preds):
            # Extract the label and prediction
            pos_user = pos[si][0,:] # user position
            label_pos = pos[si][1:nl[si].int() + 1,:]   # label position
            
            seen += 1   # number of processed images
            pred_para = pred[:,:2]

            # Evaluate
            # For training mode, if the prediction is empty, just continue
            if pred_para.size(0) < 1:
                continue

            pred_pos = para2pos(pred_para, pos_user)
            f1score, pd, rmse = cal_metric_pos(label_pos, pred_pos, detect_thres=detect_thres)
            
            iter_f1score.update(f1score)
            iter_pd.update(pd)
            iter_rmse.update(rmse)
            
            rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
            pbar.set_description(('%15s' * 1 + '%15.4g' * 4) %
                                    (f'Validation', iter_f1score.avg, iter_pd.avg, rmse_avg, imgsz))
            
    # Print results
    logger.info(f'Epoch: {epoch_id}')
    logger.info(('%15s' * 3) % ('F1-score', 'Pd', 'RMSE'))
    rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
    logger.info(('%15.4f' * 3) % tuple([iter_f1score.avg, iter_pd.avg, rmse_avg]))            
    
    # Return results
    model.float()  # for training
    rmse_avg = torch.sqrt(iter_rmse.avg) if isinstance(iter_rmse.avg, torch.Tensor) else math.sqrt(iter_rmse.avg)
    return iter_f1score.avg, iter_pd.avg, rmse_avg


def parse_opt():
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]  # root directsory
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

    parser = argparse.ArgumentParser()
    # basic parameters
    parser.add_argument('--data-path', type=str, default='./dataset', help='path to dataset')
    parser.add_argument('--pretrained', type=str, default=ROOT / 'yolov5s.pt', help='pretrained model path')
    parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=64, help='train, val image size (pixels)')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--Ns-max', type=int, default=5, help='number of maximum scatters' )
    # running parameters
    parser.add_argument('--device', type=int, help='cuda device, i.e. 0 or 0,1,2,3 or cpu') # TODO
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # technique parameters
    parser.add_argument('--detect-thres', type=float, default=1, help='detected threshold')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/config.yaml', help='model.yaml path')
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    # Directories
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    logger_dir = Path(opt.save_dir) / 'log.txt'
    logger.set_file(logger_dir)
    
    # Checks
    print_args(vars(opt))
    
    # DDP mode
    device = init_device(gpu=opt.device)
    
    # load model
    model_path = opt.pretrained
    ckpt = torch.load(model_path, map_location=device)

    model = create_model(opt.cfg or ckpt['model'].yaml).to(device)
    csd = ckpt['model'].float().state_dict()
    model.load_state_dict(csd, strict=False) 
    
    # load data
    [test_loader] = create_Env_dataloader(path=opt.data_path,
                                        batch_size=opt.batch_size,
                                        num_workers=opt.workers,
                                        device=device,
                                        Ns_max=opt.Ns_max,
                                        val_only=True)
    with torch.no_grad():
        run(dataloader=test_loader, 
            save_dir=opt.save_dir, 
            model=model.to(device),
            device=device,
            batch_size=opt.batch_size,
            imgsz=opt.imgsz,
            epoch_id=-1,
            detect_thres=opt.detect_thres)
        

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)