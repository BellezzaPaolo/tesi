import firedrake as fd
import matplotlib.pyplot as plt

def save_uh(mesh,uh, filename):
    chk = fd.CheckpointFile(filename, "w")
    chk.save_mesh(mesh)
    chk.save_function(uh, name="groundstate")
    chk.close()


def load_ground_truth(filename):
    with fd.CheckpointFile(filename, "r") as afile:
        mesh = afile.load_mesh()

        u = afile.load_function(mesh, name="groundstate")
        afile.close()

    return mesh, u
