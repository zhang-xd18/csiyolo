import argparse
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
PROJECT_ROOT = ROOT.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # Model & hyperparameters
    parser.add_argument('--cfg', type=str, default=ROOT / 'configs/config.yaml', help='model.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'configs/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--Ns-max', type=int, default=5, help='maximum number of scatters' )
    
    # Training Basic configs
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--data-path', type=str, default='sd_dataset', help='path to dataset')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=64, help='train, val image size (pixels)')
    parser.add_argument('--device', type=int, default=0, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--project', default=PROJECT_ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--val-freq', type=int, default=1, help='the frequency for validation')
    parser.add_argument('--test-freq', type=int, default=1, help='the frequency for test')
    # Training technique parameters
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')    
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--detect-thres', type=float, default=1, help='detect threshold')
    
    # Training mode
    parser.add_argument('--resume', action='store_true', help='continue training from pretrained model')
    parser.add_argument('--pretrained', type=str, default=ROOT / 'yolov5s.pt', help='pretrained model path')
    parser.add_argument('--noise-bound', type=float, default=0, help='the upper bound of the addition noise')
    
    return parser.parse_known_args()[0] if known else parser.parse_args()

