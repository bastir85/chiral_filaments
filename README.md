Modifications of LAMMPS used to generate the results of the paper: "Chiral-Filament Self-Assembly on Curved Manifolds".
Simulation is based upon the LAMMPS version "31 May 2019".
The elastic potential, enforcing bending and twist, is implemented as a pair potential in the files pair_bending.cpp and pair_bending.h.

The python script `create.py` generates the LAMMPS input and data files based upon the `in.channel.tmpl` file and the desired simulation parameters.

Steps to run simulations:
----

1. Download and Setup LAMMPS, follwing the offical documentation (https://lammps.sandia.gov/doc/)
1. Copy the modified/added files into the LAMMPS `src` folder
1. Compile desired LAMMPS version
1. Create input and data file with python3.6+ `python create.py`
1. Run simualtion `$LAMMPS_BIN -in in.channel`


[![DOI](https://zenodo.org/badge/281688627.svg)](https://zenodo.org/badge/latestdoi/281688627)
