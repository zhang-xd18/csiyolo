import yaml
import torch
import os
import inspect
import contextlib
import time
from typing import Optional
from pathlib import Path
from utils import logger

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def clustering(prediction, conf_thres=5e-1, dis_thres=1.5, max_nms=100):
    """ Clusters inference results to filter out nearby or overlapping detections.
    Args:
        prediction (torch.Tensor or list/tuple of torch.Tensor): The model's prediction output, expected to be a tensor 
            of shape (batch_size, n, 3) where each detection contains [x, y, confidence].
        conf_thres (float, optional): Confidence threshold for selecting candidates. Defaults to 0.25.
        dis_thres (float, optional): Distance threshold for merging nearby detections. Defaults to 1.
        max_nms (int, optional): Maximum number of detections to consider after sorting by confidence. Defaults to 100.
        out_thres (float, optional): Confidence threshold for the final output detections. Defaults to 0.1.
    Raises:
        AssertionError: If `conf_thres` are not within the range [0, 1].
    Notes:
        The circular boundary correction is specific to cases where the x-coordinate wraps around a fixed boundary (e.g., 0 to 64).
    Returns:
        list of detections, on (n, 3) tensor per image [x, y, conf]
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device

    # select candidates
    xc = prediction[:, 2, :] > conf_thres  # candidates
    preds = []
    # iteration over images
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.T
        x = x[xc[xi]]  # select candidates
        if not x.shape[0]:  # no detections
            preds.append(torch.empty(0, 3, device=device))
            continue
        # sort by confidence and limit to max_nms
        x = x[x[:, 2].argsort(descending=True)[:max_nms]]

        # Calculate pairwise distances
        xy = x[:, :2]
        dist = torch.cdist(xy, xy, p=2)

        # Mask for distances greater than threshold
        mask = dist > dis_thres

        # Correction for circular situation
        circular_mask = torch.ones_like(mask)
        near_boundary = (abs(x[:, 0]) < 1) | (abs(64. - x[:, 0]) < 1)
        if near_boundary.any():
            cx = x[:, 0].clone()
            cx[abs(x[:, 0]) < 1] += 64.
            circular_dist = torch.cdist(torch.stack((cx, x[:, 1]), dim=1), xy, p=2)
            circular_mask = circular_dist > dis_thres
        combined_mask = mask & circular_mask

        # merge nearby points
        out = torch.zeros_like(x)
        out[0] = x[0]
        for i in range(1, x.shape[0]):
            if torch.all(combined_mask[i,:i]):
                out[i] = x[i]
            else:
                out[torch.where(combined_mask[i,:i] == False)[0][0], 2] += x[i, 2]
                
        # Post process
        out = out[abs(out[:,1]) > 1e-6] # remove zero points
        out[:,0] += 1.   # correct for the delay
        
        out = out[out[:,0] > 0]
        out[:, 2] /= out[:, 2].sum()    # normalize confidence     
        out_th = out[:,2].mean() * 1/3 # 0.6 * out.shape[0] / x.shape[0]   
        preds.append(out[out[:, 2] > out_th])  # threshold confidence for output        
    return preds





class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0, device: torch.device = None):
        """Initializes a profiling context for YOLOv5 with optional timing threshold and device specification."""
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Logs the arguments of the calling function, with options to include the filename and function name."""
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    logger.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    """Validates if a file or files have an acceptable suffix, raising an error if not."""
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def yaml_save(file="data.yaml", data=None):
    """Safely saves `data` to a YAML file specified by `file`, converting `Path` objects to strings; `data` is a
    dictionary.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def colorstr(*input):
    """
    Colors a string using ANSI escape codes, e.g., colorstr('blue', 'hello world').

    See https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]