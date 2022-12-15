import os
import pathlib
import torch
from mesh import TriangleMesh
from tets import TetGrid, Vertex
from torch.utils.data import Dataset
from tqdm import tqdm
from igl import point_mesh_squared_distance
import numpy as np

from multiprocessing import Manager
import concurrent.futures
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

## We should be able to batch this through torch instead of parallelizing with IGL,
## which would be faster
def process_mesh(inp):
    fname, namespace = inp
    out_dir = namespace.out_dir
    calculate_deformation = namespace.calculate_deformation
    aug = namespace.aug
    verts = namespace.verts
    centroids = namespace.centroids

    print(f'Processing mesh {fname}')
    mesh = TriangleMesh.from_file(str(fname))
    # random axis-aligned scaling augmentations
    if aug:
        if np.random.uniform(0, 1) > 0.1:
            random_scaling = torch.FloatTensor(3).uniform_(0.85, 1.15)
            for vertex in mesh.vertices:
                vertex.coord *= random_scaling

    v = torch.stack([x.coord for x in mesh.vertices])
    center = v.mean(dim=0)
    mesh.translate(center.neg())
    
    scale = 0.5 / torch.stack([x.coord for x in mesh.vertices]).abs().amax()
    mesh.scale(scale)

    mesh.translate(torch.tensor([.5, .5, .5]))

    o = [Vertex(x, y, z) for x, y, z in centroids]

    occ = mesh.compute_winding_nums(o)
    occ = occ.round()

    if torch.all(torch.logical_or(occ == 1, occ == 0)):
        # In the few cases where this happens, just truncate
        occ[occ < 0] = 0
        occ[occ > 1] = 1

    if calculate_deformation:
        # vector from vertex to closest point
        v = torch.stack([x.coord for x in mesh.vertices]).numpy()
        f = np.array([[np.array(x.index) for x in f.vertices] for f in mesh.faces])
        d, _, c = point_mesh_squared_distance(verts, v, f)
        deformations = torch.stack(
            [
                torch.tensor(c[idx]) - x  for idx, x in enumerate(verts)
            ]
        ).float()
        
        # vector from centroid to closest point
        d2, _, c2 = point_mesh_squared_distance(centroids, v, f)
        deformations_c = torch.stack(
            [
                torch.tensor(c2[idx]) - x for idx, x in enumerate(centroids)
            ]
        ).float()
    
    else:
        deformations = torch.empty(1)
        deformations_c = torch.empty(1)

    mesh_out_path = pathlib.Path(out_dir) / fname.stem
    os.makedirs(mesh_out_path)
    torch.save(occ.float(), mesh_out_path / "occ.pt")
    torch.save(deformations, mesh_out_path / "def.pt")
    torch.save(deformations_c, mesh_out_path / "def_c.pt")
    
    return occ.sum(), deformations.abs().amax(), deformations_c.abs().amax()


def load_data(
    mesh_dir: str, out_dir: str, 
    grid: TetGrid, calculate_deformation: bool = True,
    aug: bool = False,
    num_workers: int = 1):
    directory = pathlib.Path(mesh_dir)
    files = list(directory.glob("*"))
    print(f"Reading {len(files)} meshes")

    if calculate_deformation:
        verts = torch.stack([v.coord for v in grid.vertices]).numpy()
        centroids = torch.stack([t.centroid() for t in grid.tetrahedrons]).numpy()     

    mgr = Manager()
    ns = mgr.Namespace()
    ns.out_dir = out_dir
    ns.calculate_deformation = calculate_deformation
    ns.aug = aug
    ns.verts = verts
    ns.centroids = centroids

    print("Computing features")
    if num_workers == 1:
        results = [process_mesh(fname.resolve(), ns) for fname in tqdm(files)]
    else:
        with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
            results = list(tqdm(executor.map(process_mesh, [(fname.resolve(), ns) for fname in files]), total=len(files)))

    pos_weight = sum([o for o, _, _ in results]) / len(files) / len(grid.tetrahedrons)
    # scalar so that maximum deformation is 1
    deformation_scalar = 1 / max(torch.stack([d for _, d, _ in results]).amax(), torch.stack([d_c for _, _, d_c in results]).amax())
    
    torch.save(pos_weight, os.path.join(out_dir, 'pos_weight.pt'))
    torch.save(deformation_scalar, os.path.join(out_dir, 'deformation_scalar.pt'))


class MeshFeatureDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
    ):
        self.data_dir = pathlib.Path(data_dir)
        self.meshes = [p.stem for p in self.data_dir.glob('*') if p.is_dir()]
        self.pos_weight = torch.load(self.data_dir / 'pos_weight.pt')
        self.deformation_scalar = torch.load(self.data_dir / 'deformation_scalar.pt')
        
    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, index):
        path = self.data_dir / self.meshes[index]
        ret = (
            self.meshes[index],
            torch.load(path / 'occ.pt'),
            torch.load(path / 'def.pt'),
            torch.load(path / 'def_c.pt'),
        )
        
        return ret


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', help='Directory of mesh files to process', type=str)
    parser.add_argument('--out_dir', help='Directory to write processed data to', type=str)
    parser.add_argument('--initial_grid', help='File of initial coarse grid', type=str)
    parser.add_argument('--subdivision_depth', help='number of times to subdivide grid', type=int)
    parser.add_argument('--augmentations', help='Set to do random axis aligned scaling augmentations', action='store_true')
    parser.add_argument('--num_workers', help='Number of parallel workers for data processing', type=int, default=1)

    args = parser.parse_args()
    grid = TetGrid.from_file(args.initial_grid)
    for _ in range(args.subdivision_depth):
        grid = grid.subdivide()
    load_data(args.mesh_dir, args.out_dir, grid, True, args.augmentations, args.num_workers)


if __name__ == '__main__':
    main()
