import os
import numpy as np
from scipy.io import savemat, loadmat
import time
from .generate_utils import Show_map, theta2phi, Cal_beam, Cal_delay, Post_process

class DataConfig:
    def __init__(self, distance=100, c=3e8, Nt=64, Nc=1024, df=100e3, Ncut=64, beta=10 ** (-1), f0=28e9):
        self.distance = distance
        self.c = c
        self.Nc = Nc
        self.df = df 
        self.Ncut = Ncut
        self.Nt = Nt
        self.beta = beta
        self.f0 = f0

    def as_dict(self):
        return {
            'distance': float(self.distance),
            'c': float(self.c),
            'Nt': int(self.Nt),
            'Nc': int(self.Nc),
            'df': float(self.df),
            'Ncut': int(self.Ncut),
            'beta': float(self.beta),
            'f0': float(self.f0),
        }

def generate_map(Ns, N_UE=1, config=DataConfig(), visualize_flag=False):
    
    distance, c, Nc, df, Ncut = config.distance, config.c, config.Nc, config.df, config.Ncut
    
    p_BS = np.array([0, 0])
    p_UE = np.random.rand(N_UE, 1) * distance
    p_UE = np.hstack((p_UE, np.random.rand(N_UE, 1) * distance - distance / 2))
    p_s = np.random.rand(Ns, 1) * distance
    p_s = np.hstack((p_s, np.random.rand(Ns, 1) * distance - distance / 2))
    
    for i in range(N_UE):
        p_UE_i = p_UE[i, :]  # position of UE

        ######################   Check user distribution #########################
        sin_theta_UE = p_UE_i[1] / np.linalg.norm(p_UE_i)
        while sin_theta_UE > np.sqrt(3)/2 or sin_theta_UE < -np.sqrt(3)/2:
            new_UE_x = np.random.rand(1, 1) * distance
            new_UE_y = np.random.rand(1, 1) * distance - distance / 2
            p_UE[i, :] = np.hstack((new_UE_x, new_UE_y))
            p_UE_i = p_UE[i, :]
            sin_theta_UE = p_UE_i[1] / np.linalg.norm(p_UE_i)

        tau = (np.linalg.norm(p_s - p_BS, axis=1) + np.linalg.norm(p_s - p_UE_i, axis=1) - np.linalg.norm(p_UE_i - p_BS)) / c
        p_t = p_s - p_BS
        sin_theta = p_t[:, 1] / np.linalg.norm(p_t, axis=1)
        
        
        ######################   Check scatter distribution #########################
        while any(tau < 1 / (Nc * df)) or any(tau * df * Nc / Ncut > 1) or any(sin_theta > np.sqrt(3)/2) or any(sin_theta < -np.sqrt(3)/2) or any(abs(np.arcsin(sin_theta) - np.arcsin(sin_theta_UE)) < 1/180 * np.pi):
            # Show_map(p_BS, p_UE, p_s, distance, True)
            fail_delay = np.logical_or(tau < 1 / (Nc * df), tau * df * Nc / Ncut > 1)
            fail_theta = np.logical_or(np.logical_or(sin_theta > np.sqrt(3)/2, sin_theta < -np.sqrt(3)/2), abs(np.arcsin(sin_theta) - np.arcsin(sin_theta_UE)) < 1/180 * np.pi)
            fail_point = np.logical_or(fail_delay, fail_theta)

            num_fail = np.sum(fail_point)
            new_x = np.random.rand(num_fail, 1) * distance
            new_y = np.random.rand(num_fail, 1) * distance - distance / 2
            p_s[fail_point, :] = np.hstack((new_x, new_y))
            tau = (np.linalg.norm(p_s - p_BS, axis=1) + np.linalg.norm(p_s - p_UE_i, axis=1) - np.linalg.norm(p_UE_i - p_BS)) / c
            p_t = p_s - p_BS
            sin_theta = p_t[:, 1] / np.linalg.norm(p_t, axis=1)    
  
    ##################### Visulization #########################
    Show_map(p_BS, p_UE, p_s, distance, visualize_flag)
    return p_BS, p_UE, p_s

def generate_SU_channel(p_BS, p_UE, p_s, Ns, config=DataConfig(), saveraw=False):
    
    Nc, Nt, Ncut, c, df, f0, beta = config.Nc, config.Nt, config.Ncut, config.c, config.df, config.f0, config.beta

    # Channel generate
    H = np.zeros((Nc, Nt), dtype=np.complex128)
    NL = np.zeros((1))
    Gain = np.zeros((Ns + 1), dtype=np.complex128)
    P_s = np.zeros((Ns + 1, 2))
    Para_s = np.zeros((Ns + 1, 2))

    # determine the number of activated scatters/links
    NL = Ns  # TODO: currently the fixed number of scatters
    index = np.sort(np.random.permutation(Ns)[:NL])  # random select activated scatters
    p_s_i = np.vstack((p_UE, p_s[index, :]))
    P_s[:NL + 1, :] = p_s_i  # store positions of activated scatters

    # Calculate tau
    tau = (np.linalg.norm(p_s_i - p_BS, axis=1) + np.linalg.norm(p_s_i - p_UE, axis=1) - np.linalg.norm(p_UE - p_BS)) / c
    


    # Calculate theta
    p_t = p_s_i - p_BS
    sin_theta = p_t[:, 1] / np.linalg.norm(p_t, axis=1)

    # store parameters
    phi = theta2phi(sin_theta)  # compute the normalized theta --> phi
    t = tau * df * Nc / Ncut # compute the normalized tau --> t
    Para_s[:NL + 1, :] = np.vstack((t, phi)).T

    # Steering vector
    beam_vec = Cal_beam(sin_theta, Nt)
    # Delay vector
    delay = Cal_delay(tau, Nc, df, f0)

    # Complex gain
    g_LOS = np.random.randn() + 1j * np.random.randn()
    g_NLOS = (np.random.randn(NL) + 1j * np.random.randn(NL)) * beta
    g = np.hstack((g_LOS, g_NLOS))
    Gain = g


    # Frequency-Antenna domain Channel matrix
    H = delay.T @ np.diag(g) @ beam_vec.T

    return H, P_s, Para_s, Gain, NL

def generate_dataset(N_sample, dir, name, Ns, saveraw=False, *, config=None):
    start = time.time()
    ##############################  Parameter setting   #########################
    config = config or DataConfig()
    N_UE = 1  # number of users
    Nc, Nt, Ncut = config.Nc, config.Nt, config.Ncut
    Ns = int(Ns) if isinstance(Ns, str) else Ns  # ensure Ns is an integer


    ##############################   Storage prepare  #########################
    H_single = np.zeros((N_sample, Nc, Nt), dtype=np.complex128)
    HS_single = np.zeros((N_sample, 2 * Ncut * Nt), dtype=np.double)
    HA_single = np.zeros((N_sample, 1 * Ncut * Nt), dtype=np.double)
    P_single = np.zeros((N_sample, Ns + 1, 2))
    Gain_single = np.zeros((N_sample, Ns + 1), dtype=np.complex128)
    NL_single = np.zeros((N_sample, 1))
    Para_single = np.zeros((N_sample, Ns + 1, 2))

    ##############################   Data generation   #########################
    for n in range(N_sample):

        ##############################   Generate map   #########################
        p_BS, p_UE, p_s = generate_map(Ns, N_UE, config, visualize_flag=False)
        
        ######################   Generate single user channel #########################
        H_i, P_i, Para_i, Gain_i, NL_i = generate_SU_channel(p_BS, p_UE, p_s,
                                                             Ns=Ns,
                                                             config=config,
                                                             saveraw=saveraw)

        H_single[n, :, :] = H_i
        P_single[n, :, :] = P_i
        Gain_single[n, :] = Gain_i
        NL_single[n] = NL_i
        Para_single[n, :, :] = Para_i
        
        ##################   Post process ###################
        hs, ha = Post_process(H_i, Nt, Ncut)
        HS_single[n, :] = hs
        HA_single[n, :] = ha
        
        if n % 1000 == 0:
            print(f'loaded {name} Ns {Ns}, {n} samples, time: {time.time() - start:.2f}s')
            start = time.time()

    ######################  Storage  #########################
    savedir = os.path.join(dir, f"Ns_{Ns}")
    os.makedirs(savedir, exist_ok=True)
    
    # Save the data
    print('start saving')
       
    if saveraw:
        savemat(
            os.path.join(savedir, f"DATA_{name}.mat"),
            {
                'H': H_single,
                'P': P_single,
                'Para': Para_single,
                'Gain': Gain_single,
                'NL': NL_single,
                'meta': config.as_dict(),
            },
        )
    
    savemat(
        os.path.join(savedir, f"TDATA_{name}.mat"),
        {
            'HS': HS_single,
            'HA': HA_single,
            'P': P_single,
            'Para': Para_single,
            'Gain': Gain_single,
            'NL': NL_single,
            'meta': config.as_dict(),
        },
    )
    
    print(f'finished saving, time: {time.time() - start:.2f}s')