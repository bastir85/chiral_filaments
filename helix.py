import numpy as np
import pandas as pd
from sklearn.neighbors import DistanceMetric, NearestNeighbors

from utils import load_xyz
import collections

frames_cache = {}

def get_cluster_growth(data):
    def calc_clusters(df):
        cluster_sizes = df.iloc[0].value_counts()
        a = cluster_sizes.value_counts()
        a.index = [f"N_{i}" for i in a.index]
        
        N = cluster_sizes.sum()
        part_ava_cluster_size = (cluster_sizes**2).sum()/N
     
        cluster_sizes_part = [t for t in cluster_sizes for _ in range(t)]
        p25, p75 = np.percentile(cluster_sizes_part, [25, 75])

        part_ava_cluster_size_std = np.sqrt((cluster_sizes*(cluster_sizes - part_ava_cluster_size)**2).sum()/N)
        b = pd.Series({"cluster_size_average": cluster_sizes.mean(), "cluster_size_std":cluster_sizes.std(),
                       "cluster_size_min":cluster_sizes.min(), "cluster_size_max":cluster_sizes.max(),
                       "part_ava_cluster_size": part_ava_cluster_size, "part_ava_cluster_size_std":part_ava_cluster_size_std,
                       "cluster_size_p25": p25, "cluster_size_p75": p75})
        return b.combine_first(a)

    return data.xs(2,level=1).groupby(level=0).apply(calc_clusters).unstack().fillna(0.0)


def build_pbc_diff(Lx, Ly):
    def _pbc_diff(x,y):
        d = x-y
        return np.array([d[0] - np.round(1.*d[0]/Lx)*Lx, 
                d[1] - np.round(1.*d[1]/Ly)*Ly])
    return _pbc_diff    



def load_last_frame(wdir):
    lastF = frames_cache.get(wdir)
    if lastF is None:
        data = load_xyz(open(wdir+"/clusters.xyz"))
        lastF = data.loc[data.last_valid_index()[0]].T
        frames_cache[wdir] = lastF
    return lastF

def read_lammps_parameters(in_file):
    parameters = {}
    with open(in_file) as fp:
        header = [next(fp) for x in range(12)]
    parameters["atoms"] = int(header[2].split(" ")[0])
    parameters["atom_types"] = int(header[3].split(" ")[0])
    parameters["bonds"] = int(header[4].split(" ")[0])
    parameters["bond_types"] = int(header[5].split(" ")[0])
    parameters["angles"] = int(header[6].split(" ")[0])
    parameters["angle_types"] = int(header[7].split(" ")[0])
    
    parameters["sizex"] = float(header[9].split(" ")[1])
    parameters["sizey"] = float(header[10].split(" ")[1])
    parameters["sizez"] = float(header[11].split(" ")[1])
    return parameters


def unwrap(angles):
    'Unwraps such that mean is correct for a (-pi/2,+pi/2)'
    if len(angles)% 2 == 0:
        angles = angles.copy()[1:]
    else:
        angles = angles.copy()
    angles = angles.copy()
    mean_angle = np.median(angles) 
    mean_angle = mean_angle + np.pi % (2*np.pi) - np.pi
    m1 = angles - mean_angle > np.pi/2
    m2 = angles - mean_angle < -np.pi/2
    angles[m1] -= np.pi
    angles[m2] += np.pi
    return angles

def angle_hist(lastF, parameters):
    pbc_diff = build_pbc_diff(Lx = parameters["sizex"],Ly = parameters["sizey"])
    pbc_metric = lambda x, y: np.sqrt(np.sum(pbc_diff(x,y)**2))

    cluster_sizes = lastF[2].value_counts()
    cids = cluster_sizes.sort_values(ascending=False)
    clusters = {}
    algo = NearestNeighbors(2, metric=pbc_metric)
    for cid, csize in list(cids.iteritems()):
        cluster = lastF.loc[lastF[2] == cid]
        algo.fit(cluster[[0,1]].values)
        b = algo.radius_neighbors(radius=1.1)[1]

        orients = []
        degrees = []
        
        for idx, bb in enumerate(b):
            p0 = cluster.iloc[idx]
            degrees.append(len(bb))
            for bbb in bb:
                p1 = cluster.iloc[bbb]
                vec = pbc_diff(p0, p1)
                orients.append([str(sorted([p0.name, p1.name])), np.arctan(vec[1]/ vec[0]), (vec[0], vec[1])])
        if orients: 
            clusters[cid] = {"orients": pd.DataFrame(orients).drop_duplicates(subset=[1]), 
                             "degrees":collections.Counter(degrees)}

    return clusters

def post_process(df):
    wdir = df.name
    data = load_xyz(open(wdir+"/clusters.xyz"))
    lastF = data.loc[data.last_valid_index()[0]].T
    parameters = read_lammps_parameters(wdir+"/data.channel")

    angles = angle_hist(lastF, parameters)
    cluster_growth = get_cluster_growth(data)    
    return angles, cluster_growth
