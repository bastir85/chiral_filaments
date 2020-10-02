import os


def load_xyz(filename):
    import numpy as np
    import pandas as pd
    data_xyz = _iter_loadxyz(filename)
    tsteps, frames = zip(*data_xyz)
    frames = np.array(frames)
    frames = frames.swapaxes(1,2).reshape(-1,frames.shape[1])
    idx = pd.MultiIndex.from_product([tsteps,[0,1,2]], names=['time', 'coordinate'])
    data = pd.DataFrame(frames, index=idx)
    return data

