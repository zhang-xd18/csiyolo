import torch
import torch.nn as nn

__all__ = ["ComputeLoss"]


class ComputeLoss:
    # Compute losses
    def __init__(self, model, autobalance=False):
        """Initializes ComputeLoss with model and autobalance option, autobalances losses if True."""
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))
        MSEpos = nn.MSELoss()  # position loss

        self.balance = [4.0, 1.0, 0.4]
        self.BCEobj, self.MSEpos, self.gr, self.hyp, self.autobalance, self.nl, self.device = \
            BCEobj, MSEpos, 1.0, h, autobalance, 3, device


    def __call__(self, p, targets):
        """Performs forward pass, calculating localization and objectness loss for given predictions and targets."""
        
        lpos = torch.zeros(1, device=self.device)  # position loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tpos, indices = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gi, gj = indices[i]  # image, gridy, gridx
            
            tobj = torch.zeros_like(pi[:,0,:,:], dtype=pi.dtype, device=self.device).unsqueeze(1)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, _ = pi[b, :, gi, gj].split((2, 1), 1)  # predicted xy
                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5   # convert relative x,y to -0.5~1.5
                dis = torch.sum((pxy - tpos[i]) ** 2, dim=1, keepdim=True)
                lpos += dis.mean()
                
                # Objectness
                dis = dis.detach().clamp(0).type(tobj.dtype)    
                tobj[b, :, gi, gj] = 1.
                    
                
            obji = self.BCEobj(pi[:,2,...].unsqueeze(1), tobj)
            lobj += obji * self.balance[i]  # obj loss
        lpos *= self.hyp["pos"]
        lobj *= self.hyp["obj"]
        bs = tobj.shape[0]  # batch size

        return (lpos + lobj) * bs, torch.cat((lpos / self.hyp["pos"], lobj / self.hyp["obj"])).detach()

    def build_targets(self, p, targets):
        """Prepares model targets from input targets (image,x,y) for loss computation, returning positions and indices.
        """
        
        nt = targets.shape[0]  # number of targets
        tpos, indices = [], []
        
        gain = torch.ones(3, device=self.device)  # normalized to gridspace gain
        g = 0.5  # bias
        off = (torch.tensor([[0, 0], 
                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=self.device).float() * g)  # offsets


        for i in range(self.nl):
            shape = p[i].shape
            gain[1:3] = torch.tensor(shape)[[3, 2]]

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Define
                gxy = t[:,1:3]
                gxi = gain[[1, 2]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            b, gxy = t[:,0], t[:,1:3]
            b = b.long()
            gij = (gxy - offsets).long()
            gi, gj = gij.T
            
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tpos.append(gxy - gij)
            
        return tpos, indices