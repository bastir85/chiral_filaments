#create pool data file

import random
import os
from math import sqrt,pi
import numpy as np
from time import sleep

DIR = os.path.join(os.getcwd(), "special")


def init_sim(eps, kb, kt, alpha0, packing_fraction=0.6, R=50):
    b0 = np.sin(alpha0)**2/R
    t0 = np.sin(alpha0)*np.cos(alpha0)/R
    seed = int(np.random.randint(1e6))
    # Create WDIR
    name='vary_strength{}_a0{}_b{}t{}_pf{}_R{}_s{}'.format(eps,alpha0, kb, kt, packing_fraction, R, str(seed)[-2:])
    wdir=os.path.join(DIR,name)
    os.makedirs(wdir)
    sleep(1)
   
    #Parameters from in.channel.tmpl
    #------------------
    bl=0.8
    d1=d2=0.9
    #------------------
    dy=2*d1 #use outer atoms for size
    dx=2*(bl+d1)

    NPM=3; #Atoms per molecule
    atom_types=2;
    bond_types=1;
    angle_types=1;
    box_boundary_y=int(np.round(2*np.pi*R))
    box_boundary_x=int(np.round(25*R))
    N=packing_fraction*box_boundary_x*box_boundary_y/(dx*dy)

    U=box_boundary_y
    min_dist=2.0
    ratio=1.0*box_boundary_x/box_boundary_y
    Nx = np.round(np.sqrt(N*ratio))
    Ny = np.round(np.sqrt(N/ratio))
    N=int(Nx*Ny)
    if box_boundary_x/Nx < min_dist:
        print("Error box to small!")
        Nx = box_boundary_x/min_dist
    if box_boundary_y/Ny < min_dist:
        print("Error box to small!")
        Ny = box_boundary_y/min_dist
    X=np.linspace(0, box_boundary_x, Nx, endpoint=False)
    Y=np.linspace(0, box_boundary_y, Ny, endpoint=False)
    number_of_bonds=N*(NPM-1);
    print('Creating pool of ' + str(N) +' molecules and aspect ratio ' + str(box_boundary_x/box_boundary_y) + '\n');

    output_file='data.channel';

    fo=open(wdir+'/'+output_file,"w");

    #Header

    header= 'LAMMPS channel simulation\n\n' + \
    str(N*NPM)+' atoms \n' + \
    str(atom_types)+ ' atom types \n' + \
    str(number_of_bonds)+' bonds \n' + \
    str(bond_types)+ ' bond types \n' + \
    str(N)+' angles \n' + \
    str(angle_types)+ ' angle types \n';

    fo.write(header+' \n');

    #Box
    Box='0 '+str(box_boundary_x)+' xlo xhi \n' + \
    '0 '+str(box_boundary_y)+' ylo yhi \n' + \
    '-0.1 0.1 zlo zhi \n';

    fo.write(Box+' \n');

    #Masses
    Masses='Masses \n\n';

    for i in range(1,atom_types+1):
        Masses+=str(i)+ ' 1 \n';
    fo.write(Masses+' \n');

    #Atoms
    fo.write('Atoms \n\n');

    molid=1
    z=0.0;
    for x in X:
        for y in Y:
            px=random.random();
            py=random.random();
            norm=sqrt(px*px+py*py);
            px=bl/norm*px; py=bl/norm*py;
            atype=1; #atom type	
            fo.write('%d %d %d %g %g %g \n' % (molid*NPM-2, molid,1,x,y,z));
            fo.write('%d %d %d %g %g %g \n' % (molid*NPM-1, molid,2,x+px,y+py,z));
            fo.write('%d %d %d %g %g %g \n' % (molid*NPM, molid,1,x+2*px,y+2*py,z));
            molid+=1


    #Bonds
    fo.write('\nBonds \n\n');

    #only harmonic bonds
    atom_n=1;
    for bond in range(1,N*(NPM-1)+1,2):
        btype=1; #bond type
        atom1=atom_n;
        atom2=atom1+1;
        fo.write('%d %d %d %d\n' % (bond, btype,atom1,atom2));
        atom1=atom_n+1;
        atom2=atom1+1;
        fo.write('%d %d %d %d\n' % (bond+1, btype,atom1,atom2));
        atom_n+=3;
    fo.write(' \n');

    #Angles
    fo.write('\nAngles \n\n');

    atom_n=1; #including sponetnous curvature
    for angle in range(1,N+1):
        atype=1; #angle type
        atom1=(angle-1)*3+1;
        atom2=atom1+1;
        atom3=atom1+2;
        fo.write('%d %d %d %d %d\n' % (angle, atype,atom1,atom2,atom3));

    fo.write('\n');
    fo.close();

    dir_path = os.path.dirname(os.path.realpath(__file__))
    fo = open(dir_path+"/in.channel.tmpl")
    tmpl = fo.read()
    fo.close()
    tmpl = tmpl.replace("{{EPS}}", str(eps))
    tmpl = tmpl.replace("{{b0}}", str(b0))
    tmpl = tmpl.replace("{{t0}}", str(t0))
    tmpl = tmpl.replace("{{kappa_b}}", str(kb))
    tmpl = tmpl.replace("{{kappa_t}}", str(kt))
    tmpl = tmpl.replace("{{R}}", str(R))
    tmpl = tmpl.replace("{{seed}}", str(seed))
    fo = open(wdir+"/in.channel", "w" )
    fo.write(tmpl)
    fo.close()
    fo = open(wdir+"/parameters", "w" )
    fo.write("espilon\t{}\n".format(eps))
    fo.write("bend_rigidity\t{}\n".format(kb))
    fo.write("twist_rigidity\t{}\n".format(kt))
    fo.write("packing_fraction\t{:.2f}\n".format(N*dx*dy/box_boundary_x/box_boundary_y))
    fo.write("b0\t{}\n".format(b0))
    fo.write("t0\t{}\n".format(t0))
    fo.write("R\t{}\n".format(R))
    fo.write("alpha0\t{}".format(alpha0))
    fo.close()
R=30.
for strength in [0,1e-5,1e-4,1e-3,1e-2,1e-1,5e-1,1,2,3,4,5]:
    init_sim(eps=22, kb=strength*R**2, kt=strength*R**2, alpha0=36./180.*np.pi, packing_fraction=0.7, R=R)

