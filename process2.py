# coding: utf-8

import helix
import pandas as pd
import pickle
import os
from multiprocessing import Pool, Manager, Lock

n_jobs = 24*2 

with pd.HDFStore("results.h5") as store:
    results = store.get("v5")
#print(len(results))
#results = results.loc[results.index.duplicated(keep='first')]
#print(len(results))

def process_single(sim):
    wdir, df = sim
    print(f"Process {wdir}")
    angles, cluster_growth = helix.post_process(df)
    wdir2 = "/".join(wdir.split("/")[:-1])
    os.makedirs(f"postprocess/{wdir2}", exist_ok=True) 
    with open(f"postprocess/{wdir}.pickle", "wb") as fp:
         pickle.dump({'angles':angles, 'cluster_growth': cluster_growth}, fp)
         
     
mask = [not os.path.exists(f"postprocess/{name}.pickle") for name in results.index]
sims =  list(results[mask].iterrows())

print(len(sims))
with Pool(n_jobs) as p:
    p.map(process_single, sims)
