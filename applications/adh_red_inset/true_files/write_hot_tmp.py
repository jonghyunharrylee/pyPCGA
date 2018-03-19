#! /usr/bin/env python

import os
from enkf_tools import Mesh_lite

def write_constant_free_surface(base_3dm,z_f,t0=0.0,suffix='3dm'):
     assert os.path.isfile(base_3dm+'.'+suffix)
     mesh        = Mesh_lite.Mesh2DM(base_3dm,suffix=suffix)
     output_file = base_3dm+'.hot'
     with open(output_file,'w') as fout:
          fout.write("""DATASET
OBJTYPE "mesh2d"
BEGSCL
ND {0}
NC {1}
NAME "IOH"
TIMEUNITS seconds
TS {2}\n""".format(mesh.nNodes_global,mesh.nElements_global,t0))
          for I in range(mesh.nNodes_global):
              fout.write("{0:10.8f}\n".format(z_f-mesh.nodeArray[I,2]))
          fout.write("ENDDS\n")
     
if __name__=='__main__':
     
     import argparse
     parser = argparse.ArgumentParser(description="Write hotstart file")

     parser.add_argument("--orig_mesh","-f",
                        help="full name for xms mesh",
                        action="store",
                        default=None)
     parser.add_argument("--z_f",
                        help="Constant free-surface elevation",
                        action="store",
                        type=float,
                        default=0.0)
     parser.add_argument("--t0",
                         help="Constant free-surface elevation",
                         action="store",
                         type=float,
                         default=0.0)
    

     opts = parser.parse_args()

     assert os.path.isfile(opts.orig_mesh),"{0} input mesh not found".format(opts.orig_mesh)
 
     ##open the original mesh
     base3dm=os.path.basename(opts.orig_mesh)
     dir3dm =os.path.dirname(opts.orig_mesh)
     suffix='3dm'
     name  = base3dm
     if len(base3dm.split('.')) > 1:
          suffix=base3dm.split('.')[1]
          name  = base3dm.split('.')[0]

        
     write_constant_free_surface(name,opts.z_f,opts.t0,suffix=suffix)
     
