# coding: utf-8

from sr.numpyc import load_xyz
import numpy as np
import pandas as pd
import glob
import csv
import os
H5FP = 'results.h5'

def exists(wdir):
    try:
        with pd.HDFStore(H5FP) as store:
            results = store.get("v5") 
        df = results.loc[wdir]
    except KeyError:
        return False
    print ("--skip--")
    return True

def find_crossings(data, y0, eps = 0.8):
    lastF = data.loc[data.last_valid_index()[0]][:2].T
    lastF.columns = ["x","y"]
    candidates = lastF[(lastF.y > y0 - eps) & (lastF.y < y0 + eps)].copy()

    candidates["cid"] = -1
    candidates["dmin"] = 999
    for idx1, (index1,c1) in enumerate(candidates.iterrows()):
        for idx2, (index2,c2) in enumerate(candidates.iloc[idx1+1:].iterrows()):
            dn = np.linalg.norm(c1[["x","y"]]-c2[["x","y"]])
            index = index1 if index1 < index2 else index2
            if dn > 1.1:
                continue
            if candidates.loc[index2,"dmin"]  > dn:
                candidates.loc[index2,"dmin"] = dn
                candidates.loc[index2,"cid"] = index


            if candidates.loc[index1,"dmin"]  > dn:
                candidates.loc[index1,"dmin"] = dn
                candidates.loc[index1,"cid"] = index
    return candidates


def process(dir):
    if not os.path.exists(dir + "/parameters"):
        print(f"{dir} is empty!")
        return

    r = csv.reader(open(dir+"/parameters"), delimiter="\t")
    parameters = dict(r)
    for key,val in parameters.items():
        parameters[key] = float(val)
    R = parameters["R"] 
    data = load_xyz(open(dir+"/clusters.xyz"))
    cid = data.xs(2,level=1).iloc[-1]
    cluster_sizes = cid.value_counts()
    N = cluster_sizes.sum()
    cluster_sizes_part = [t for t in cluster_sizes for _ in range(t)]
    p25, p75 = np.percentile(cluster_sizes_part, [25, 75])

    part_ava_cluster_size = (cluster_sizes**2).sum()/N
    part_ava_cluster_size_std = np.sqrt((cluster_sizes*(cluster_sizes - part_ava_cluster_size)**2).sum()/N)
    print(part_ava_cluster_size, np.mean(cluster_sizes_part),p25, p75)

    bins = np.arange(0,N,10)
    H,bi=np.histogram(cluster_sizes.loc[cid],bins=bins, normed=False)
    hdata = np.array([(bi[1:]+bi[:-1])/2.,1.0*H/N])

    parameters.update(cluster_size_average=cluster_sizes.mean(),
        cluster_size_std=cluster_sizes.std(), cluster_size_min=cluster_sizes.min(), cluster_size_max=cluster_sizes.max(),
        part_ava_cluster_size=part_ava_cluster_size, part_ava_cluster_size_std=part_ava_cluster_size_std, R=R,
        cluster_size_p25=p25, cluster_size_p75=p75
        #y_crossings_mean=c.mean(), y_crossings_std=c.std(),
        )
    parameters["epsilon"] = parameters["espilon"]
    del parameters["espilon"]
    if 'alpha0' in parameters:
       del parameters["alpha0"]
    df = pd.DataFrame(parameters, index=(dir,))
    df.to_hdf(H5FP, 'v5', append=True, min_itemsize = {'index': 128})

    df_hist = pd.DataFrame(hdata.T, columns=["cluster_size_w10","num_particles"])
    df_hist.to_hdf(H5FP, "hdata/{}".format(dir))
    return df

if __name__ == "__main__":

    for dir in glob.glob("*/*"):
        print("Process " + dir)
        if exists(dir):
            continue
        
        df = process(dir)
        print(df)
