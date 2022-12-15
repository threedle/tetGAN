import numpy as np
import torch
from igl import winding_number
from tets import Face, TetGrid, Vertex, Tetrahedron
from tqdm import tqdm

import meshio


class TriangleMesh:
    def __init__(self, vertices: list[Vertex], faces: list[Face], grid_to_surface=None):
        self.vertices = vertices
        self.faces = faces
        self.grid_to_surface = grid_to_surface  # dictionary of vertex coordinate to surface mesh index
        if self.grid_to_surface is not None:
            self.grid_idx_to_surface = {k.index: v for k, v in grid_to_surface.items()}
        if self.grid_to_surface is not None:    # grid_to_surface is set when a surface mesh is extracted
            self.surface_to_grid = TriangleMesh.reverse_mapping(grid_to_surface)
        else:
            self.surface_to_grid = None

        #self.edges = self.get_edges()

    @classmethod
    def from_file(cls, path: str, disable_progress: bool = False):
        vertices, faces = TriangleMesh.load_mesh(path, disable_progress)
        return cls(vertices, faces)

    @staticmethod
    def reverse_mapping(grid_to_surface):
        surface_to_grid = {}
        for v in grid_to_surface.keys():
            surface_to_grid[grid_to_surface[v]] = v.index  # key: vertex index in extracted surface, val: vertex index
        return surface_to_grid

    def get_edges(self):
        edges = []
        for f_id, face in enumerate(self.faces):
            for i in range(3):
                edge = (face.vertices[i].index, face.vertices[(i + 1) % 3].index)
                edge = tuple(sorted(edge))
                if edge not in edges:
                    edges.append(edge)
        return edges  # unique edges

    def get_edge_faces(self):
        edge_faces = {}  # for each edge, which faces contain it
        for f_id, face in enumerate(self.faces):
            for i in range(3):
                edge = (face.vertices[i].index, face.vertices[(i + 1) % 3].index)
                edge = tuple(sorted(edge))
                edge_faces[edge] = edge_faces.get(edge, []) + [f_id]
        return edge_faces

    def get_vert_edges(self):
        vert_edges = {}  # for each vertex, which edges contain it
        for v in self.vertices:
            for e in self.edges:
                if v.index in e:  # e is a tuple of vert indices
                    vert_edges[v.index] = vert_edges.get(v.index, []) + [e]
        return vert_edges

    def get_vert_faces(self):
        vert_faces = {}  # for each vertex, which faces contain it (by face index)
        for f_id, face in enumerate(self.faces):
            for i in range(3):
                v_idx = face.vertices[i].index
                vert_faces[v_idx] = vert_faces.get(v_idx, []) + [f_id]

        return vert_faces

    @classmethod
    def from_winding_nums_and_grid(cls, winding_nums: torch.Tensor, grid: TetGrid, disable_progress: bool = False, threshold: float = 0.5):
        winding_nums = winding_nums.flatten()
        vertices = {}
        faces = []

        for tet in (grid.tetrahedrons if disable_progress else tqdm(grid.tetrahedrons)):
            if winding_nums[tet.index] > 0.5 and len(
                    tet.boundary_vertices) > 0:  # if occupied and has more than 1 boundary vert
                for v in tet.boundary_vertices:  # for all boundary vertices
                    if v not in vertices:
                        vertices[v] = len(vertices)
                face = Face(tet.boundary_vertices, winding_nums[tet.index])  # face is original face

                face.orient(tet.internal_vertices[0], True)
                faces.append(face)

            for neighbor in tet.neighbors:
                if neighbor.dummy:
                    continue
                if neighbor.index < tet.index:
                    continue
                if (winding_nums[neighbor.index] > 0.5) ^ (winding_nums[tet.index] > 0.5):
                    #print(winding_nums[neighbor.index], winding_nums[tet.index])
                    shared_vertices = []
                    for v in tet.vertices:
                        for v1 in neighbor.vertices:
                            if v.index == v1.index:
                                shared_vertices.append(v)

                    assert len(shared_vertices) == 3

                    if winding_nums[tet.index] > 0.5:
                        external_vertex = (set(neighbor.vertices) - set(shared_vertices)).pop()
                        in_prob_contr = winding_nums[tet.index]
                        out_prob_contr = 1 - winding_nums[neighbor.index]
                    else:
                        external_vertex = (set(tet.vertices) - set(shared_vertices)).pop()
                        in_prob_contr = winding_nums[neighbor.index]
                        out_prob_contr = 1 - winding_nums[tet.index]

                    prob = (in_prob_contr + out_prob_contr) / 2
                    if prob > threshold:
                        for v in shared_vertices:
                            if v not in vertices:
                                vertices[v] = len(vertices)  # key: vertex (index, coord), val: extracted surface index
                        face = Face(shared_vertices, prob)

                        face.orient(external_vertex, False)
                        faces.append(face)

        rebound_vertices = [Vertex(x, y, z, vertices[v]) for (x, y, z), v in [(v.coord, v) for v in vertices]]
        rebound_faces = [Face([rebound_vertices[i] for i in [vertices[v] for v in f.vertices]], f.prob) for f in faces]

        return cls(rebound_vertices, rebound_faces, grid_to_surface=vertices)

    @staticmethod
    def load_mesh(path: str, disable_progress: bool = False) -> tuple[list[Vertex], list[Face]]:
        """load triangular mesh from file"""
        ext = path.split('.')[-1]
        vertices = []
        faces = []

        with open(path, "r") as input_file:
            if not disable_progress:
                print(f"Reading .{ext} file")

            if ext == 'obj':
                idx = 0
                for line in input_file:
                    if line[:2] == 'v ':
                        _, x, y, z = line.split(' ')
                        vertices.append(Vertex(float(x), float(y), float(z), idx))
                        idx += 1

                    # assume faces come after all vertices in .obj file
                    elif line[0] == 'f':
                        indices = [s.split('/')[0] for s in filter(lambda x: not (x.isspace() or x == ''), line.split(' '))]
                        faces.append(Face([vertices[int(i) - 1] for i in indices[1:]]))
                    else:
                        continue

            elif ext == 'off' or ext == 'ply':
                num_vertices = -1
                num_faces = -1
                line = ''
                if ext == 'ply':
                    while line != 'end_header\n':
                        line = input_file.readline()
                        info = line.split(' ')
                        if info[0] == 'element':
                            if info[1] == 'vertex':
                                num_vertices = int(info[2])
                            elif info[1] == 'face':
                                num_faces = int(info[2])
                elif ext == 'off':
                    line = input_file.readline()
                    if line == 'OFF\n':
                        line = input_file.readline()
                    num_vertices, num_faces, _ = map(int, line.split(' '))

                assert num_vertices > 0
                assert num_faces > 0

                if not disable_progress:
                    print(f"Reading {num_vertices} vertices")
                for i in range(num_vertices) if disable_progress else tqdm(range(num_vertices)):
                    line = input_file.readline()
                    coordinates = line.split(' ')[:3]
                    vertices.append(Vertex(*[round(float(c), ndigits=5) for c in coordinates], i))

                if not disable_progress:
                    print(f"Reading {num_faces} faces")
                for i in range(num_faces) if disable_progress else tqdm(range(num_faces)):
                    line = input_file.readline()
                    vertex_indices = line.split(' ')[1:4]
                    curr_vertices = [vertices[int(j)] for j in vertex_indices]
                    faces.append(Face(curr_vertices))
            else:
                raise ValueError('unsupported file type')
            
            if not disable_progress:
                print("Finished reading")

            ## enable print
            # sys.stdout = sys.__stdout__
            return vertices, faces

    def scale(self, scale):
        for v in self.vertices:
            v.scale(scale)

    def translate(self, translate: torch.Tensor):
        for v in self.vertices:
            v.translate(translate)

    def nonnegative(self):
        """translate so all vertices are nonnegative"""
        vec = torch.min(torch.amin(torch.stack([v.coord for v in self.vertices]), dim=0), torch.Tensor([0])).neg()
        self.translate(vec)

    def compute_winding_nums(self, points: list[Vertex]) -> torch.Tensor:
        v = torch.stack([v.coord for v in self.vertices]).numpy()
        f = np.array([[v.index for v in f.vertices] for f in self.faces])
        o = torch.stack([v.coord for v in points]).numpy()
        return torch.tensor(winding_number(v, f, o))

    # Deformation weighted laplace smoothing
    # in-place operation
    def laplace_smoothing(self, pred_defs):
        num_vertices = len(pred_defs)
        n_tens = [[v.index] for v in self.vertices]
        for src, sink in self.get_edges(): 
            n_tens[src].append(sink)
            n_tens[sink].append(src)
        n_tens_grid = list(map(lambda x: list(map(lambda y: self.surface_to_grid[y], x)), n_tens))
        counts = torch.tensor([len(l) for l in n_tens_grid])
        max_len = counts.amax()

        n_tens = torch.tensor([
            l + [len(self.vertices) for _ in range(max_len - len(l))] for l in n_tens
        ])
        n_tens_grid = torch.tensor([
            l + [num_vertices for _ in range(max_len - len(l))] for l in n_tens_grid
        ])
        temp = torch.nn.functional.normalize(
            torch.cat([pred_defs, torch.zeros(1, 3)], dim=0)[n_tens_grid, :],
            p=2, dim=-1
        )
        weights = (temp[:, 0].unsqueeze(1).expand(len(temp), max_len, 3) * temp).sum(dim=-1).abs()
        weights[:, 0] *= 0.5
        weights = torch.nn.functional.normalize(weights, p=1)
        
        coords = torch.cat([torch.stack([v.coord for v in self.vertices]), torch.zeros(1, 3)], dim=0)
        new_mesh_coords = (coords[n_tens, :] * weights.unsqueeze(-1)).sum(dim=1)
        for i, v in enumerate(self.vertices):
            v.coord = new_mesh_coords[i]

    def write_to_obj(self, file=None, filepath: str = None):
        if file == None and filepath == None:
            raise ValueError("Either a file object or a filepath must be provided")

        with (open(filepath, 'w') if file is None else file) as f:
            for v in self.vertices:
                x, y, z = v.coord.numpy()
                f.write(f"v {round(x.item(), 5)} {round(y.item(), 5)} {round(z.item(), 5)}\n")

            for face in self.faces:
                f.write(f"f {' '.join(str(s) for s in [v.index + 1 for v in face.vertices])}\n")

    def write_to_off(self, file=None, filepath: str = None):
        if file == None and filepath == None:
            raise ValueError("Either a file object or a filepath must be provided")

        with (open(filepath, 'w') if file is None else file) as f:
            f.write('OFF\n')
            f.write(f'{len(self.vertices)} {len(self.faces)} 0\n')
            for v in self.vertices:
                x, y, z = v.coord.numpy()
                f.write(f"{round(x.item(), 5)} {round(y.item(), 5)} {round(z.item(), 5)}\n")

            for face in self.faces:
                f.write(f"3 {' '.join(str(s) for s in [v.index for v in face.vertices])}\n")


class TetMesh:
    def __init__(self, vertices: list[Vertex], faces: list[Face], tets: list[Tetrahedron], mesh_to_grid = None):
        self.vertices = vertices
        self.faces = faces
        self.tets = tets
        self.mesh_to_grid = mesh_to_grid
        self.face_verts_indices = sorted(set(
            [v.index for f in self.faces for v in f.vertices]
        ))
    
    @classmethod
    def from_winding_nums_and_grid(
        cls, winding_nums: torch.Tensor, grid: TetGrid, disable_progress: bool = False,
        defs = None, ls = 0
    ):
        winding_nums = winding_nums.flatten()
        occ_indices = (winding_nums > 0.5).nonzero().flatten()
        occ_tets = [grid.tetrahedrons[i] for i in occ_indices]

        if not disable_progress:
            print("Creating surface mesh")

        surface_mesh = TriangleMesh.from_winding_nums_and_grid(winding_nums, grid, disable_progress)
        for _ in range(ls):
            surface_mesh.laplace_smoothing(defs)
        
        req_v_grid_indices = sorted(set([v.index for t in occ_tets for v in t.vertices]))
        mesh_to_grid = dict(zip([i for i in range(len(req_v_grid_indices))], req_v_grid_indices))
        grid_to_mesh = dict(zip(req_v_grid_indices, [i for i in range(len(req_v_grid_indices))]))
        vertices = []
        faces = []
        tets = []
        for i, idx in mesh_to_grid.items():
            vertices.append(Vertex(
                *(grid.vertices[idx].coord if idx not in surface_mesh.grid_idx_to_surface else surface_mesh.vertices[surface_mesh.grid_idx_to_surface[idx]].coord), 
                index=i)
            )
        
        for face in surface_mesh.faces:
            faces.append(Face([vertices[grid_to_mesh[surface_mesh.surface_to_grid[v.index]]] for v in face.vertices]))
        
        for i, tet in enumerate(occ_tets):
            tets.append(Tetrahedron([vertices[grid_to_mesh[v.index]] for v in tet.vertices], i))

        return cls(vertices, faces, tets, mesh_to_grid)
    
    def write(self, output_path):
        vertices = torch.stack([v.coord for v in self.vertices])
        tets = torch.stack([torch.tensor([v.index for v in tet.vertices]) for tet in self.tets])
        meshio.Mesh(vertices, [("tetra", tets)]).write(output_path)


        


