/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */


#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_bending.h"
#include "atom.h"
#include "atom_vec.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "neigh_list.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "random_mars.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairBending::PairBending(LAMMPS *lmp) : Pair(lmp)
{
  nmax = 0;
}

/* ---------------------------------------------------------------------- */

PairBending::~PairBending()
{

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
  }


}

/* ---------------------------------------------------------------------- */

void PairBending::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double delx,dely,delz,evdwl,fpair;
  double rsq,dot1,dot2,dot3,ran_num;
  int *ilist,*jlist,*numneigh,**firstneigh;
  tagint moli,molj;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  double *special_lj = force->special_lj;
  int newton_bond = force->newton_bond;
  int newton_pair = force->newton_pair;
  tagint *molecule = atom->molecule;
  bigint natoms = atom->natoms;


  // grow local arrays if necessary
  // need to be atom->nmax in length

  if (atom->nmax > nmax) {
    nmax = atom->nmax;
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double A,B,C;
  // loop over neighbors of my atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    moli = molecule[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      jtype = type[j];
      molj = molecule[j];
      if (moli == molj) continue;

      delx = x[j][0] - x[i][0];
      dely = x[j][1] - x[i][1];
      delz = x[j][2] - x[i][2];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq > cutsq[itype][jtype]) continue;
      A = (1 + pow(M_E,8 - 8*rsq))*(1 - pow(M_E,32 - 8*rsq))*(-2*delx*dely*k_b*(-(k_0*cyl_r*rsq) + pow(dely,2)) + k_t*(delx*dely - cyl_r*rsq*t_0)*(pow(dely,2) - pow(delx,2)));
      B = 8*rsq*(1 + pow(M_E,24))*pow(M_E,8 - 8*rsq)*(k_t*pow(-(delx*dely) + cyl_r*rsq*t_0,2) + k_b*pow(-(k_0*cyl_r*rsq) + pow(dely,2),2));
      C = 2*pow(M_E,-24)*pow(cyl_r,-2)*pow(rsq,-3)*pow(1 + pow(M_E,8 - 8*rsq),-2);

      f[i][0] += -C *(A*dely + B*delx);
      f[i][1] += +C*(A*delx - B*dely);
      if (newton_pair || j < nlocal) {
        f[j][0] += +C*(A*dely + B*delx); 
        f[j][1] += -C*(A*delx - B*dely);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairBending::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(cut,n+1,n+1,"pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairBending::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }


}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairBending::coeff(int narg, char **arg)
{
  if (narg < 7 || narg > 8)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  k_b = force->numeric(FLERR,arg[2]);
  k_t = force->numeric(FLERR,arg[3]);
  cyl_r = force->numeric(FLERR,arg[4]);
  k_0 = force->numeric(FLERR,arg[5]);
  t_0 = force->numeric(FLERR,arg[6]);

  double cut_one = cut_global;
  if (narg == 8) cut_one = force->numeric(FLERR,arg[7]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairBending::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBending::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBending::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairBending::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairBending::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairBending::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g\n",i,j,cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairBending::single(int i, int j, int itype, int jtype, double rsq,
                        double factor_coul, double factor_lj,
                        double &fforce)
{
  double r,arg,philj;

  r = sqrt(rsq);
  arg = MY_PI*r/cut[itype][jtype];
  fforce = factor_lj * sin(arg) * MY_PI/cut[itype][jtype]/r;
  return factor_lj*philj;
  //return force and energy!
}
