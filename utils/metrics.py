import torch

NCUT = 64
NT = 64
NC = 1024
DF = 100e3
c = 3e8

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision1, precision2
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float())   # precision
    precision_2 = torch.mean((dist2 < threshold).float())   # recall
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2 


def para2pos(pred_para, pos_user):
    # recover tau
    tau = pred_para[:, 0]
    tau = tau / NC / DF

    # recover theta
    phi_theta = pred_para[:, 1]
    transfer = lambda x: x * 2 / NT if x <= NT/2 else (x / NT - 1) * 2
    sin_theta = torch.tensor(list(map(transfer, phi_theta)), device=pred_para.device)

    # calculate positions
    d = torch.norm(pos_user, p=2)
    cos_alpha, sin_alpha = pos_user[0]/d, pos_user[1]/d
    cos_theta = torch.sqrt(1 - sin_theta ** 2)
    cos_beta = cos_theta * cos_alpha + sin_theta * sin_alpha
    a = (c * tau * (2 * d + c * tau)) / (2 * (d + c * tau - d * cos_beta) + 1e-10)  # avoid nan
    
    y = a * sin_theta
    x = a * cos_theta
    pred_pos = torch.stack((x,y), dim=1)
    
    # adjust for the error in identifying the LOS
    if (tau[0] != 0) and any(tau == 0):
        pos_pred_add = pred_pos[0,:]    # the first element is NLOS and needs to be added
        
        index = torch.nonzero(tau == 0)
        pos_user_est = pred_pos[index[0], :]    # select the first estimated LOS elements
        pred_pos[index[0], :] = pos_pred_add.clone()
        pred_pos[0,:] = pos_user_est
    
    # Clamp pred_pos values to the specified ranges
    pred_pos[:, 0] = torch.clamp(pred_pos[:, 0], min=0, max=100)
    pred_pos[:, 1] = torch.clamp(pred_pos[:, 1], min=-50, max=50)
    
    return pred_pos


def pair_distance(A, B):
    """
    Compute the pair Distance between two point clouds.

    Args:
        A (torch.Tensor): True Point cloud A of shape (N1, 2).
        B (torch.Tensor): Point cloud B of shape (N2, 2).

    Returns:
        torch.Tensor: The Pair Distance between the two point clouds.
    """
    # Ensure the tensors are on the same device
    A = A.to(B.device)

    # Compute pairwise squared distances
    dist_mat = torch.cdist(A, B, p=2) ** 2

    # Compute the Chamfer distance
    dist_A_to_B = torch.min(dist_mat, dim=1)[0] # (N1,)
    dist_B_to_A = torch.min(dist_mat, dim=0)[0] # (N2,)
    
    return dist_A_to_B, dist_B_to_A


def cal_metric_pos(label_pos, pred_pos, detect_thres=1):
    '''
    compute the false detection rate
    input: label_pos: Ns * 2 (x, y)
           pred_pos: Ns * 2 (x, y)
           threshold: float
    output: f_score, probability of detection, rmse
    '''
    dist1, dist2 = pair_distance(label_pos, pred_pos)    
    rmse = dist1[dist1 < detect_thres]
    f_score, pd, _ = fscore(dist1, dist2, threshold=detect_thres)
    return f_score, pd, rmse
    

class AverageMeter(object):
    r"""Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self, name):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.name = name

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if val.numel() == 0:
            return
        self.val = val
        self.sum += val.sum()
        self.count += val.numel()
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"==> For {self.name}: sum={self.sum}; avg={self.avg}"

