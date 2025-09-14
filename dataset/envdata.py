import os
import scipy.io as sio
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

__all__ = ['create_Env_dataloader']


def create_Env_dataloader(path, batch_size, num_workers, device, Ns_max=5, val_only=False):
    output_dataloader = []
    filename = ['TDATA_train', 'TDATA_val', 'TDATA_test'] if not val_only else ['TDATA_test']
    for name in filename:
        dataset = EnvDataset(data_dir=os.path.join(path, name+'.mat'), 
                             device=device, Ns_max=Ns_max)
        
        dataloader = DataLoader(dataset=dataset, 
                                batch_size=batch_size, 
                                shuffle=True if 'train' in name else False, 
                                pin_memory=True,
                                num_workers=num_workers,
                                collate_fn=env_collate_fn)
        output_dataloader.append(dataloader)
    return output_dataloader

class EnvDataset(Dataset):
    def __init__(self, data_dir, device, Ns_max=5):
        print(data_dir)
        
        self.device, self.channel, self.nt, self.nc, self.Ns_max = \
            device, 2, 64, 64, Ns_max        

        # data loading
        HA, Pos, Para, NL = self._pack(data_dir)
        self.data_dic = {"H": HA,
                        "Para": Para,
                        "Pos": Pos,
                        "NL": NL}
        
        
    def _pack(self, dir):
        import mat73
        try: 
            matfile = mat73.loadmat(dir)
        except:
            matfile = sio.loadmat(dir)

        H = matfile['HS']
        H = torch.tensor(H, dtype=torch.float32).view(
            H.shape[0], self.nt, self.nc, self.channel)
        # NOTE: adjust due to the reshape in the data generation, to be optimized in the future
        H = H.permute(0, 3, 2, 1)   
        # NOTE: Only use the real and imag part of the channel 
        HA = H

        Pos = matfile['P']
        Pos = torch.tensor(Pos, dtype=torch.float32).view(
            Pos.shape[0], self.Ns_max + 1, 2)

        Para = matfile['Para']
        Para = torch.tensor(Para, dtype=torch.float32).view(
            Para.shape[0], self.Ns_max + 1, 2)
        Para = Para[:,1:,:]
        Para = torch.cat([torch.zeros(Para.shape[:2]).unsqueeze(2), Para], dim=2)
        # NOTE: Adjust for the delay, remove the first of LOS path.
        Para[:,:,1] -= 1/self.nc    
        
        
        NL = matfile['NL']
        NL = torch.tensor(NL, dtype=torch.float32).view(
            NL.shape[0], 1)
        
        return ([HA, Pos, Para, NL])    # (original channel, position, parameter, #scatter)
    
    def __getitem__(self, idx):
        ''' Get the image and the target '''
        ''' image: CWH '''
        ''' return values are formatted for YOLO valuation '''
        ''' output: channel, parameter, position'''
        
        h = self.data_dic['H'][idx,:,:,:]
        para = self.data_dic['Para'][idx,:,:]
        pos = self.data_dic["Pos"][idx, :, :]
        nl = self.data_dic["NL"][idx, :]

        return h, para, pos, nl


    def __len__(self):
        return len(self.data_dic['H'])
    

def env_collate_fn(batch):
    im, para, pos, nl = zip(*batch)
    for i, lb in enumerate(para):
        lb[:, 0] = i  # add target image index for build_targets()
    return torch.stack(im, 0), torch.cat(para, 0), torch.stack(pos, 0), torch.stack(nl, 0) 
