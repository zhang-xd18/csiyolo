# Modified from train.py in original YOLOv5, by zhang-xd18
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
import time
import yaml
import torch
import thop
import numpy as np
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch.optim import lr_scheduler
from utils.loss import ComputeLoss
from utils.parser import parse_opt
from utils.init import init_device
from dataset import create_Env_dataloader
from models import create_model
from utils import logger
from utils.general import colorstr, yaml_save, check_suffix, increment_path, print_args
from utils.torch_utils import smart_optimizer, one_cycle
import val as validate
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format

def add_noise(channel, noise_std_max):
    bs, ch, nt, nc = channel.shape
    noise_std = torch.rand(bs, 1, device=channel.device) * noise_std_max
    noise_matrix = torch.randn(bs, ch, nt, nc, device=channel.device) * noise_std.view(bs, 1, 1, 1)
    nmse = 10 * torch.log10(torch.sum(noise_matrix ** 2, dim=(1, 2, 3)) / torch.sum(channel ** 2, dim=(1, 2, 3)))
    noisy_channel = channel + noise_matrix
    
    return noisy_channel, torch.mean(nmse)


def train(hyp, device, opt):
    '''Main function for training'''
    
    # Hyperparameters
    data_path, save_dir, epochs, batch_size, resume, pretrained, workers = \
        Path(opt.data_path), Path(opt.save_dir), opt.epochs, opt.batch_size, opt.resume, opt.pretrained, opt.workers
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best_f = w / 'last.pt', w / 'best_f.pt' # last, best model
    if logger._log_file == None:
        logger.set_file(save_dir / "log.txt")
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()
    yaml_save(save_dir / 'hyp.yaml', hyp)
    yaml_save(save_dir / 'opt.yaml', vars(opt))
    shutil.copy(Path(opt.cfg), save_dir / 'config.yaml')
    
    # Basic settings
    imgsz = opt.imgsz  # image sizes

    # Model
    check_suffix(pretrained, '.pt')  # check weights    
    if resume & str(pretrained).endswith('.pt'): # resume from an existing pretrained model
        ckpt = torch.load(pretrained, map_location=device)
        model = create_model(opt.cfg or ckpt['model'].yaml).to(device)  # create
        csd = ckpt['model'].float().state_dict()
        model.load_state_dict(csd, strict=False)  # load      
        logger.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {pretrained}')  # report
    else:
        model = create_model(opt.cfg).to(device)  # create
    print(model)

    # Print model & FLOPs
    p = next(model.parameters())
    im = torch.empty((1, 2, imgsz, imgsz), device=p.device)
    flops, params = thop.profile(deepcopy(model), inputs=(im,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")
    logger.info(f"Model Summary: {len(list(model.modules()))} layers, {params} Params, {flops} FLOPs")

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  

    # Load Dataset
    train_loader, val_loader, test_loader = create_Env_dataloader(path=data_path,
                                                    batch_size=batch_size,
                                                    num_workers=workers,
                                                    device=device,
                                                    Ns_max=opt.Ns_max)


    # Model attributes
    model.hyp = hyp  # attach hyperparameters to model
    best_fscore, start_epoch = 0, 0
    
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    scheduler.last_epoch = start_epoch - 1
    compute_loss = ComputeLoss(model)  # init loss class

    logger.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    
    logger.info(f'Using linear mode to add noise with upper bound {opt.noise_bound}.')
    noise_func = lambda epoch: epoch / epochs * opt.noise_bound
    
    # Train
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        mloss = torch.zeros(2, device=device)  # mean losses
        pbar = enumerate(train_loader)
        
        logger.info(('\n' + '%15s' * 5) % ('Epoch', 'pos_loss', 'obj_loss', 'Instances', 'Size'))
        
        pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        optimizer.zero_grad()

        # Update noise std
        noise_std_max = noise_func(epoch)

        for i, (imgs, targets, _, nl) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float()
            imgs, _ = add_noise(imgs, noise_std_max)  # add noise

            targets = targets[targets[:,1] > 0, :]
            assert targets.shape[0] == nl.sum(), f"number of targets do not match, check dataset"
            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Forward            
            pred = model(imgs)  # forward
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size

            
            # Backward
            torch.use_deterministic_algorithms(False)
            optimizer.zero_grad()
            loss.backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                optimizer.step()
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            pbar.set_description(('%15s' * 1 + '%15.4g' * 4) %
                                    (f'{epoch}/{epochs - 1}', *mloss, targets.shape[0], imgs.shape[-1]))
            
            if i % (nb / 10) == 0:
                logger.info('\n')
                logger.info(f"Epoch {epoch}/{epochs} | Batch {i}/{nb}: noise std: {noise_std_max:.2e}, position loss: {mloss[0]:.4g}, obj loss:{mloss[1]:.4g} \n")
            # end batch ------------------------------------------------------------------------------------------------
        scheduler.step()
        results = 0 # for debug
        final_epoch = (epoch + 1 == epochs)
        if epoch % opt.val_freq == 0 or final_epoch:
            with torch.no_grad():
                f1score, _, _ = validate.run(dataloader=val_loader,
                                            save_dir=save_dir,
                                            model=model,
                                            device=device,
                                            batch_size=batch_size,
                                            imgsz=imgsz,
                                            epoch_id=epoch,
                                            detect_thres=opt.detect_thres,
                                            )
            
            fi = f1score
            if fi > best_fscore:
                best_fscore = fi
                
            # Save model
            ckpt = {
                'epoch': epoch,
                'best_f': best_fscore,
                'model': deepcopy(model),
                'optimizer': optimizer.state_dict(),
                'opt': vars(opt),
                'date': datetime.now().isoformat()}
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fscore == fi:
                torch.save(ckpt, best_f)                            
            del ckpt
        
        if epoch % opt.test_freq == 0 or final_epoch:
            with torch.no_grad():
                f1score, _, _ = validate.run(
                                                dataloader=test_loader,
                                                save_dir=save_dir,
                                                model=model,
                                                device=device,
                                                batch_size=batch_size,
                                                imgsz=imgsz,
                                                epoch_id=epoch,
                                                detect_thres=opt.detect_thres,
                                                )
        # end epoch -------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------

    logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    for f in last, best_f:
        if f.exists():
            if f is best_f:
                logger.info(f'\Testing {f}...')
                ckpt = torch.load(f, map_location=device)
                csd = ckpt['model'].float().state_dict()
                model.load_state_dict(csd, strict=False)    
                with torch.no_grad():
                    results = validate.run(dataloader=test_loader,
                                            save_dir=save_dir,
                                            model=model.to(device),
                                            device=device,
                                            batch_size=batch_size,
                                            imgsz=imgsz,
                                            epoch_id=ckpt['epoch'],
                                            detect_thres=opt.detect_thres
                                            )

    torch.cuda.empty_cache()
    
    return results



def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    logger_dir = Path(opt.save_dir) / 'log.txt'
    logger.set_file(logger_dir)
    
    # Checks
    print_args(vars(opt))
    assert len([opt.cfg]) or len([opt.weights]), 'either --cfg or --weights must be specified'
    device = init_device(gpu=opt.device)
    
    # Train
    train(str(opt.hyp), device, opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
