import numpy as np
import time
    
def Show_map(p_BS, p_UE, p_s, distance, flag):
    if flag:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(p_BS[0], p_BS[1], marker='o', linewidths=1)
        plt.scatter(p_UE[:, 0], p_UE[:, 1], marker='x', linewidths=1)
        plt.scatter(p_s[:, 0], p_s[:, 1], marker='*', linewidths=1)
        plt.xlim([0, distance])
        plt.ylim([-distance / 2, distance / 2])
        plt.grid(True)
        plt.legend(['BS', 'UE', 'Scatters'])
        plt.show()
        plt.savefig('map.png')

def theta2phi(sin_theta):
    if isinstance(sin_theta, np.ndarray):
        phi = sin_theta.copy()
    else:
        phi = sin_theta.clone()
        
    for n in range(len(sin_theta)):
        if sin_theta[n] > 0:
            phi[n] = sin_theta[n] / 2
        elif sin_theta[n] < 0:
            phi[n] = (sin_theta[n] / 2 + 1)
        else:
            phi[n] = 0
    return phi


def phi2theta(phi):
    if isinstance(phi, np.ndarray):
        sin_theta = phi.copy()
    else:
        sin_theta = phi.clone()    
        
    for n in range(len(phi)):
        if phi[n] == 0:
            sin_theta[n] = 0
        elif phi[n] < 1 / 2:
            sin_theta[n] = phi[n] * 2
        elif phi[n] > 1 / 2:
            sin_theta[n] = (phi[n] - 1) * 2
    return sin_theta


def Cal_beam(sin_theta, Nt):
    beam_vec = np.exp(-1j * np.pi * np.arange(Nt)[:, None] * sin_theta)
    return beam_vec

def Cal_delay(tau, Nc, df, f0):
    delay = np.exp(-1j * 2 * np.pi * (np.arange(Nc) * df + f0)[:, None] * tau)
    return delay.T


def Post_process(h, Nt, Ncc):
    h = np.fft.ifft(np.fft.ifft(h, axis=0), axis=1)
    h = h.T
    absmax = np.max(np.max(np.abs(h[:, 1:Ncc + 1])))
    h_a = np.abs(h[:, 1:Ncc + 1]) / absmax  # normalized, h_a in [0,1]
    

    h = h / absmax / 2
    h_s = np.zeros((Ncc, Nt, 2))
    h_s[:, :, 0] = h[:, 1:Ncc + 1].real.T
    h_s[:, :, 1] = h[:, 1:Ncc + 1].imag.T
    
    h_a = h_a.reshape((1, Nt * Ncc))
    h_s = h_s.reshape((1, 2 * Nt * Ncc)) + 0.5

    return h_s, h_a

