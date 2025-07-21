import numpy as np
from numpy import quantile
import matplotlib.pyplot as plt

def tfr_plot(M, h=1, l=0):
# Plots a TFR in grayscale.
# INPUT
#    M     :  Synchrosqueexed TFR.
#    h     :  High quantile limit.
#    l     :  Low quantile limit.

# OUTPUT   :  Plotted TFR.
# Adapted from the MATLAB implementation by Hau-Tieng Wu into Python by Shrikant Chand (shrikant.chand@duke.edu) and Po-Ying Chen in 04.2024.

    # Quantile clipping
    q = np.quantile(M, [h, l])
    M = np.clip(M, q[1], q[0])

    # Display image
    plt.imshow(M, aspect='auto', extent=[1, M.shape[1], y[0], y[-1]], cmap='gray_r', origin='lower')
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()
