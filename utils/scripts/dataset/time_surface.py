# Time surface is also called 'surface of active events'

import numpy as np
from matplotlib import pyplot as plt

def extract_data(filename,img_size = (260,346)):
    infile = open(filename, 'r')
    ts, x, y, p = [], [], [], []
    for line in infile:
        words = line.split()
        x.append(int(words[0]))
        y.append(int(words[1]))
        ts.append(float(words[2])*10e-9)
        p.append(int(words[3]))
    infile.close()

    img_size = (260,346)

    # parameters for Time Surface
    t_ref = ts[-1]      # 'current' time
    tau = 50e-3         # 50ms

    sae = np.zeros(img_size, np.float32)
    # calculate timesurface using expotential decay
    for i in range(len(ts)):
        if (p[i] > 0):
            sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)
        else:
            sae[y[i], x[i]] = -np.exp(-(t_ref-ts[i]) / tau)
        
        ## none-polarity Timesurface
        # sae[y[i], x[i]] = np.exp(-(t_ref-ts[i]) / tau)

    return sae


if __name__ == '__main__':

    sae = extract_data('utils/scripts/dataset/0000000003.txt')

    fig = plt.figure()
    fig.suptitle('Time surface')
    plt.imshow(sae, cmap='gray')
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    plt.colorbar()
    plt.savefig('time_surface.jpg')
    plt.show()
