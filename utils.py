import firedrake as fd

def save_uh(mesh,uh, filename):
    chk = fd.CheckpointFile(filename, "w")
    chk.save_mesh(mesh)
    chk.save_function(uh, name="groundstate")
    chk.close()


def load_ground_truth(filename):
    chk = fd.CheckpointFile(filename, "r")
    mesh = chk.load_mesh()
    W = fd.FunctionSpace(mesh, "CG", 1)

    uh = fd.Function(W)
    chk.load_function(uh, name="groundstate")
    chk.close()

    return mesh, W, uh 
