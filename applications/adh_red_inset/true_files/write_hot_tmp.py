#! /usr/bin/env python

from enkf_tools import Mesh_lite
mesh = Mesh_lite.Mesh2DM('Inset_true',suffix='3dm')
z_f=4.5

with open('Inset_true.hot','w') as fout:
     fout.write("""DATASET
OBJTYPE "mesh2d"
BEGSCL
ND 10521
NC 20000
NAME "ioh"
TIMEUNITS seconds
TS 0.0\n""")
     for I in range(mesh.nNodes_global):
         fout.write("{0:10.8f}\n".format(z_f-mesh.nodeArray[I,2]))
     fout.write("ENDDS\n")
