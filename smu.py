# coding: utf-8

from numba import jit, prange
import numpy as np

@jit(nopython=True, nogil=True, fastmath=False)
def K_pure_diff(i, j):
    ii= i+1
    jj= j+1
    return 1./ii + 1./jj

@jit(nopython=True, nogil=True, fastmath=False)
def K_diff_size(i, j):
    ii= i+1
    jj= j+1
    return (ii+jj)*(1./ii + 1./jj)

@jit(nopython=True, nogil=True, fastmath=False)
def K_diff_and_pure(i, j, c0):
    ii= i+1
    jj= j+1
    return 1./(ii*jj/(ii+jj) + c0)


@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_simple(N):
    K0 = 1/3
    dN = np.zeros_like(N) 
    eps=np.e**-(22-8)

    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K0*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K0*N[j]*N[k]
        #dN[k] -= eps*k*N[k]
        #for j in range(k+1, len(N)):
        #    dN[k] += eps*2*N[j]
    return dN


@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_diff_and_pure(N, eps, c0, pfr):
    # e_eps=np.e**-eps ##18*5/6.
    e_eps=pfr*np.e**-eps  ##mit D=1/3 und rho
    dN = np.zeros_like(N) 
    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K_diff_and_pure(j, k-j-1, c0)*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K_diff_and_pure(j, k, c0)*N[j]*N[k]
        dN[k] -= e_eps*k*N[k]
        for j in range(k+1, len(N)):
            dN[k] += e_eps*2*N[j]
    return dN


@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_simple_dissintegration(N):
    K0 = 1/3
    dN = np.zeros_like(N) 
    eps=np.e**-(18-8)

    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K0*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K0*N[j]*N[k]
        dN[k] -= eps*k*N[k]
        for j in range(k+1, len(N)):
            dN[k] += eps*2*N[j]
    return dN

@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_only_diff(N):
    dN = np.zeros_like(N) 
    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K_pure_diff(j, k-j-1)*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K_pure_diff(j, k)*N[j]*N[k]
    return dN

@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_diff_size(N):
    dN = np.zeros_like(N) 
    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K_diff_size(j, k-j-1)*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K_diff_size(j, k)*N[j]*N[k]
    return dN

@jit(nopython=True, nogil=True, parallel=True, fastmath=False)
def smul_only_diff_disintegration(N):
    eps=np.e**-(22-8)
    dN = np.zeros_like(N) 
    for k in prange(len(N)):
        for j in range(k): ##was k 
            dN[k] += 0.5*K_pure_diff(j, k-j-1)*N[j]*N[k-j-1]
        for j in range(len(N)):
            dN[k] -= K_pure_diff(j, k)*N[j]*N[k]
        dN[k] -= eps*k*N[k]
        for j in range(k+1, len(N)):
            dN[k] += eps*2*N[j]
    return dN

def run_smul(kernel, steps, N=None, t0=None, dt=1e-1, 
             nd=None, writeout=100, kern_wargs={}):
    n0 = 2000#16256
    if N is not None and t0 is not None:
        print(f"continue with {t0/dt}")
        step0 = int(t0/dt)
    elif nd is not None:
        N = np.zeros(n0) # cluster sizes
        N[0] = nd
        step0 = 0
    else:
        assert False
    print(N[0])
    results = [N.copy()]
    steps = np.arange(step0, step0+int(steps))
    for idx in steps[1:]:
        N += kernel(N, **kern_wargs)*dt
        if idx % writeout == 0:
            results.append(N.copy())
            print(idx)
        if idx % 1e5 == 0:
            results_tmp = np.vstack(results)

            ic = np.arange(len(results_tmp.T))+1
            nc_ava = (ic**2*results_tmp).sum(axis=1) / (ic*results_tmp).sum(axis=1)
            nd_var = (ic*results_tmp).sum(axis=1).std()
            print(f"number density variation: {nd_var}")
            np.save('smulokowski_combined_c0_par.npy',np.vstack([steps[:writeout*len(nc_ava):writeout], nc_ava]))
    results = np.vstack(results)

    ic = np.arange(len(results.T))+1
    nc_ava = (ic**2*results).sum(axis=1) / (ic*results).sum(axis=1)
    nd_var = (ic*results).sum(axis=1).std()
    print(f"number density variation: {nd_var}")
    return steps[::writeout]*dt, results, nc_ava


def run_and_save_sim(eps, c0, rho):
    smul = run_smul(smul_diff_and_pure, 5e6, nd=rho, writeout=100, dt=1e-1,
                    kern_wargs={'c0':c0, 'eps': eps*5/6, 'pfr': 1.0}) #rho/0.08})
    np.save(f'smulokowski_combined_c0_{c0:.0f}__{eps:.0f}__rho{rho:.2f}_pfr1.0_f.npy',np.vstack([smul[0], smul[2]]))
    return smul


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rho", help="Inital trimer density")
    parser.add_argument("--eps", help="LJ interaction strength", default=18)
    parser.add_argument("--c0", help="c0", default=10)
    args = parser.parse_args()
    run_and_save_sim(c0=float(args.c0), eps=float(args.eps), rho=float(args.rho))
