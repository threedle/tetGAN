import itertools
import torch
from tqdm import tqdm
from typing import Optional
import numpy as np


class Vertex:
    def __init__(self, x: float, y: float, z: float, index: Optional[int] = None):
        self.coord = torch.tensor([x, y, z], dtype=torch.float32)
        self.index = index

    def __hash__(self):
        return self.coord.__hash__()

    # Probably unimportant, but just so we can have a consistent ordering on vertices for any usecases 
    def __ge__(self, v):
        x, y, z = self.coord
        x1, y1, z1 = v.coord
        if x < x1:
            return False
        if x > x1:
            return True
        if y < y1:
            return False
        if y > y1:
            return True
        if z < z1:
            return False
        return True

    def __lt__(self, v):
        return not (self >= v)

    def translate(self, vec):
        self.coord = self.coord + vec

    def scale(self, scalar):
        self.coord = self.coord * scalar


class Face:
    def __init__(self, vertices: list[Vertex], prob: Optional[float] = None):
        assert len(vertices) == 3
        self.vertices = vertices
        self.prob = prob

    def compute_normal(self):
        v0 = self.vertices[1].coord - self.vertices[0].coord
        v1 = self.vertices[2].coord - self.vertices[0].coord
        n = torch.cross(v0, v1)
        return torch.nn.functional.normalize(n, dim=0)

    def orient(self, vertex: Vertex, inwards: bool = True):
        n = self.compute_normal()
        ext = vertex.coord - self.vertices[0].coord
        if inwards == (n.dot(ext) > 0):
            self.vertices = [self.vertices[1], self.vertices[0], self.vertices[2]]


class Tetrahedron:
    def __init__(self, vertices: list[Vertex], index: int, prev_tet: Optional["Tetrahedron"] = None, dummy: bool = False):
        assert len(vertices) == 4
        self.vertices = vertices
        self.index = index
        self.dummy = dummy
        self.faces: list[Face] = [Face(v) for v in itertools.combinations(self.vertices, 3)]
        self.neighbors: list[Tetrahedron] = []
        self.boundary_vertices = []
        # these are just set(vertices) - set(boundary vertices)
        self.internal_vertices = []

        # the index of the previous tet that was subdivided
        self.prev_tet = prev_tet
        self.tet_prev_idx = prev_tet.index if prev_tet is not None else None

        # new indices of the tets that this tet is subdivided into
        self.next_tets: list["Tetrahedron"] = []
        self.tet_next_indices: list[int] = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def centroid(self):
        return torch.mean(torch.stack([v.coord for v in self.vertices]), dim=0)

    def compute_boundary_vertices(self):
        if len(self.neighbors) == 0:
            raise ValueError("Please compute neighbors in a grid")

        for v in self.vertices:
            face_count = 0
            for n in self.neighbors:
                if v in n.vertices:
                    face_count += 1
            if face_count < 3:
                self.boundary_vertices.append(v)
            else:
                self.internal_vertices.append(v)


class TetGrid:
    def __init__(self, vertices: list[Vertex], tetrahedrons: list[Tetrahedron], compute_dists = True):
        self.vertices = vertices
        self.tetrahedrons = tetrahedrons
        self.vertices_to_tet_map = {} ## computed in compute_neighbors
        self.compute_neighbors()
        for tet in self.tetrahedrons:
            tet.compute_boundary_vertices()

        self.compute_centroid_dists()
        if compute_dists:
            self.compute_centroid_dists()

    @classmethod
    def from_file(cls, path: str = '../grids/cube_0.05.tet', disable_progress = False):
        grid = cls([], [])
        grid.load_tets(path, disable_progress)
        grid.compute_neighbors()
        for tet in grid.tetrahedrons:
            tet.compute_boundary_vertices()

        return grid

    def load_tets(self, path: str, disable_progress = False):
        """load packing grid from a file"""
        with open(path, "r") as input_file:
            _, num_vertices, num_tets = input_file.readline().split(' ')

            if not disable_progress:
                print(f'Loading tet file\nReading {int(num_vertices)} vertices')
            for i in (tqdm(range(int(num_vertices))) if not disable_progress else range(int(num_vertices))):
                line = input_file.readline()
                coordinates = line.split(' ')
                self.vertices.append(Vertex(*[round(float(c), ndigits=5) for c in coordinates], i))

            if not disable_progress:
                print(f'Finished reading vertices\nReading {int(num_tets)} tetrahedrons')
            for i in (tqdm(range(int(num_tets))) if not disable_progress else range(int(num_tets))):
                line = input_file.readline()
                if line == '':
                    continue
                vertex_indices = line.split(' ')
                curr_vertices = [self.vertices[int(i)] for i in vertex_indices]
                new_tetrahedron = Tetrahedron(curr_vertices, len(self.tetrahedrons))
                self.tetrahedrons.append(new_tetrahedron)

    def compute_neighbors(self):
        vertices_to_tets_map = {}
        for tet in self.tetrahedrons:
            for v in tet.vertices:
                if v not in vertices_to_tets_map:
                    vertices_to_tets_map[v] = set()
                vertices_to_tets_map[v].add(tet)

        for tet in self.tetrahedrons:
            
            centroids = []
            for face in tet.faces:
                neighbor_set = set.intersection(*[vertices_to_tets_map[v] for v in face.vertices]) # find tets shared by all verts on this face
                # Either tet has no neighbor on this face or 1
                assert len(neighbor_set) == 1 or len(neighbor_set) == 2
                for neighbor in neighbor_set:
                    if neighbor != tet:
                        tet.add_neighbor(neighbor)
                        centroids.append(neighbor.centroid())

        self.vertices_to_tet_map = vertices_to_tets_map
        self.max_tets_per_v = max([len(l) for l in vertices_to_tets_map.values()]) if len(vertices_to_tets_map) > 0 else None

    def compute_centroid_dists(self):
        self.centroid_dist_list = [
            {
                tet.index: np.linalg.norm(v.coord - tet.centroid()) for tet in self.vertices_to_tet_map[v]
            } for v in self.vertices
        ]

    def compute_interpolate_dists(self):
        self.interpolate_dists = [
            [np.linalg.norm(tet.centroid() - n.centroid()) if not n.dummy else np.inf for n in [tet.prev_tet] + tet.prev_tet.neighbors] for tet in self.tetrahedrons
        ]
        for dists in self.interpolate_dists:
            while len(dists) < 5:
                dists.insert(0, np.inf)
        self.interpolate_dists = torch.tensor(self.interpolate_dists)

    def subdivide(self,  compute_dists=True, disable_progress=False):
        full_e_v_map = {}
        ret_v = self.vertices
        ret_t = []
        v_counter = len(self.vertices)
        t_counter = 0
        if not disable_progress:
            print(f"Subdividing {len(self.tetrahedrons)} tetrahedrons")
        for tet in (self.tetrahedrons if disable_progress else tqdm(self.tetrahedrons)):
            vertices = sorted(tet.vertices)
            curr_e_v_map = {}
            # add new vertices, which are midpoints of each edge
            for e in list(itertools.combinations(vertices, 2)):
                if (e[0].index, e[1].index) in full_e_v_map:
                    e_v = full_e_v_map[(e[0].index, e[1].index)]
                else:
                    x, y, z = (e[0].coord + e[1].coord) / 2
                    e_v = Vertex(x, y, z, v_counter)
                    full_e_v_map[(e[0].index, e[1].index)] = e_v
                    v_counter += 1

                curr_e_v_map[(e[0].index, e[1].index)] = e_v
            edges = list(curr_e_v_map.keys())
            for v in vertices:
                # connect each original vertex to the midpoints on the three edges it's in
                # creating 4 new tets
                tet_v = [v]

                for k in edges:
                    if v.index in k:
                        tet_v.append(curr_e_v_map[k])
                assert len(tet_v) == 4
                ret_t.append(Tetrahedron(tet_v, t_counter, tet))
                tet.next_tets.append(ret_t[-1])
                tet.tet_next_indices.append(t_counter)
                t_counter += 1

            # pick two vertices to exclude, the remaining four are the quad we will divide the octohedron on
            top = edges.pop(0)
            bottom = None
            for i, e in enumerate(edges):
                if len(set.intersection(set(top), set(e))) == 0:
                    bottom = edges.pop(i)
                    break
            assert bottom is not None

            # pick two vertices to be the diagonal to divide on
            d1 = edges.pop(0)
            d2 = None
            for i, e in enumerate(edges):
                if len(set.intersection(set(d1), set(e))) == 0:
                    d2 = edges.pop(i)
                    break
            assert d2 is not None

            # pick one of the two leftover vertices, pick one of top and bottom, and connect with the two diag vertices
            # to divide the octohedron into 4 tets
            for e1 in edges:
                for e2 in [top, bottom]:
                    vertices = [curr_e_v_map[e] for e in [e1, e2, d1, d2]]
                    ret_t.append(Tetrahedron(vertices, t_counter, tet))
                    tet.next_tets.append(ret_t[-1])
                    tet.tet_next_indices.append(t_counter)
                    t_counter += 1

        return TetGrid(ret_v + list(full_e_v_map.values()), ret_t, compute_dists)


    def get_vertex_neighbors(self):
        neighbors = [set() for _ in self.vertices]
        for tet in self.tetrahedrons:
            for v in tet.vertices:
                neighbors[v.index].update([vtx.index for vtx in tet.vertices])
        for v in self.vertices:
            neighbors[v.index].remove(v.index)

        return neighbors


    def write_to_file(self, file=None, filepath: str = None):
        if file == None and filepath == None:
            raise ValueError("Either a file object or a filepath must be provided")

        with (open(filepath, 'w') if file is None else file) as f:
            f.write(f"tet {len(self.vertices)} {len(self.tetrahedrons)}\n")
            for v in self.vertices:
                x, y, z = v.coord.numpy()
                f.write(f"{round(x.item(), 5)} {round(y.item(), 5)} {round(z.item(), 5)}\n")

            for tet in self.tetrahedrons:
                f.write(f"{' '.join([str(s) for s in [v.index for v in tet.vertices]])}\n")


class GridSequence:
    def __init__(self, init_grid: TetGrid, depth: int, compute_dists=True):
        self.grids = [init_grid]
        for _ in range(depth):
            self.grids.append(self.grids[-1].subdivide(compute_dists=compute_dists))
        for grid in self.grids[1:]:
            grid.compute_interpolate_dists()

