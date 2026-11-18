from ENM_utils import *
from packing_utils import *


class AlloNet(ENM):
    """An elastic network with strain sources (input) and targets (output)."""

    def __init__(self, net, k=None, l0=None, inid=None, outid=None,
                 input=(0.2,), output=(0.2,)):
        super().__init__(net, k, l0)
        self.input = []
        self.output = []

        if inid is None:
            for invalue in input:
                i, j = self._choose_pairs_boundary(seed=123)
                inidi, is_new = self._find_bond(i, j)
                self.input.append([inidi, invalue, self.l0[inidi]])
        else:
            assert len(inid) == len(input)
            for inidi, invalue in zip(inid, input):
                assert inidi < self.ne
                self.input.append([inidi, invalue, self.l0[inidi]])

        if outid is None:
            for outvalue in output:
                i, j = self._choose_pairs_boundary(seed=456)
                outidi, is_new = self._find_bond(i, j)
                self.output.append([outidi, outvalue, self.l0[outidi]])
        else:
            assert len(outid) == len(output)
            for outidi, outvalue in zip(outid, output):
                assert outidi < self.ne
                self.output.append([outidi, outvalue, self.l0[outidi]])

    def _find_bond(self, i, j, k_new=1.0, l0_new=None):
        """Return (index, is_new). Appends the bond at the end if absent."""
        if getattr(self, '_eidx', None) is None:
            self._eidx = {frozenset(e): n for n, e in enumerate(self.edges)}

        key = frozenset((i, j))
        if key in self._eidx:
            return self._eidx[key], False

        if l0_new is None:
            l0_new = np.linalg.norm(self.pts0[i] - self.pts0[j])

        self.graph.add_edge(i, j, length=l0_new)
        self.edges = list(self.edges) + [(i, j)]
        idx = len(self.edges) - 1
        self._eidx[key] = idx
        self.ne = len(self.edges)

        self.k = np.append(self.k, k_new)
        self.l0 = np.append(self.l0, l0_new)
        return idx, True

    def _choose_pairs_boundary(self, seed=1):
        edges = self._find_boundary_pairs()
        i, j = self._choose_pairs_by_distance(edges, seed)
        return i, j

    def _find_boundary_pairs(self):
        #min_degree = self.dim + 1 if self.dim == 2 else self.dim + 2
        min_degree=4
        vor = Voronoi(self.pts0[:, :self.dim])
        tri = Delaunay(self.pts0[:, :self.dim])
        verts = vor.vertices

        # pairs sharing a Voronoi ridge with one vertex inside and one outside the hull
        bound_pairs = []
        for nodes, ridge in zip(vor.ridge_points, vor.ridge_vertices):
            p1, p2 = nodes
            is_finite = np.array([(v >= 0) and (tri.find_simplex(verts[v]) >= 0)
                                  for v in ridge], dtype=bool)
            if is_finite.any() and not is_finite.all():
                bound_pairs.append([int(p1), int(p2)])

        # nonbonded pairs need min_degree neighbors each; bonded pairs need min_degree+1
        out = []
        for p in bound_pairs:
            bonded = self.graph.has_edge(p[0], p[1])
            need = min_degree + 1 if bonded else min_degree
            if self.graph.degree(p[0]) >= need and self.graph.degree(p[1]) >= need:
                out.append(p)
        return out

    def _choose_pairs_by_distance(self, edges, seed):
        # exclude pairs touching an existing source/target node, or a node that
        # would become floppy (degree < dim+2)
        excluded_nodes = []
        for source in self.input:
            excluded_nodes += list(self.edges[source[0]])
        for target in self.output:
            excluded_nodes += list(self.edges[target[0]])

        #min_degree = self.dim + 2
        min_degree = 4
        for i in range(self.n):
            if self.graph.degree(i) < min_degree:
                excluded_nodes.append(i)
        excluded_nodes = set(excluded_nodes)
        

        edges = [e for e in edges
                 if e[0] not in excluded_nodes and e[1] not in excluded_nodes]
        assert len(edges) > 0, "no candidate pairs left after exclusion"

        # distance of each candidate to the current sources/targets
        dmin = [1e20] * len(edges)
        for e, edge in enumerate(edges):
            for pair in list(self.input) + list(self.output):
                si, sj = self.edges[pair[0]]
                d = self._edge_edge_distance([si, sj], edge)
                if d < dmin[e]:
                    dmin[e] = d
        print(dmin)

        np.random.seed(seed)
        if len(self.input) == 0 and len(self.output) == 0:
            e = np.random.randint(len(edges))   # first one at random
        else:
            e = int(np.argmax(dmin))
        return edges[e]

    def _edge_edge_distance(self, e0, e1):
        dii = nx.shortest_path_length(self.graph, e0[0], e1[0], weight='length')
        dij = nx.shortest_path_length(self.graph, e0[0], e1[1], weight='length')
        dji = nx.shortest_path_length(self.graph, e0[1], e1[0], weight='length')
        djj = nx.shortest_path_length(self.graph, e0[1], e1[1], weight='length')
        return 0.25 * (dii + dij + dji + djj)

    def jacobian_matrix(self, current: bool = False,kin=None,kout=None,l0in=None,l0out=None):
        dim = self.dim
        assert dim == 2 or dim == 3, "Dimension must be 2 or 3"
        J = np.zeros((dim * self.n, dim * self.n), dtype=np.float64)
        pts = self.pts 
        k=self.k.copy()
        l0=self.l0.copy()
        if kin is None:
            for e,_,_ in self.input:
                k[e]=0.0
        else:
            for e,_,_ in self.input:
                k[e]=kin
        if kout is None:
            for e,_,_ in self.output:
                k[e]=0.0
        else:
           for e,_,_ in self.output:
                k[e]=kout
        if l0in is not None:
            for e,_,_ in self.input:
                l0[e]=l0in
        if l0out is not None:
            for e,_,_ in self.output:
                l0[e]=l0out
            
        for i in range(self.ne):
            u, v = self.edges[i]
            
            # Calculate distance vector and distance
            dx = pts[v] - pts[u]
            dist2 = np.dot(dx, dx)
            
            if dist2 == 0:
                continue
            
            dist = np.sqrt(dist2)
            
            ki = k[i]
            l0i = l0[i]
            
            term1 = ki * (1 - l0i / dist) 
            term2 = ki * l0i / (dist ** 3)
            
            bu = u * dim
            bv = v * dim
            
            for a in range(dim):
                for b in range(dim):
                    val = term1 * (1 if a == b else 0) + term2 * dx[a] * dx[b]
                    J[bu + a, bu + b] += val
                    J[bv + a, bv + b] += val
                    J[bu + a, bv + b] -= val
                    J[bv + a, bu + b] -= val
        
        return J
        
    def rigidity_matrix(self,current: bool = False):
        dim = self.dim
        assert dim == 2 or dim == 3, "Dimension must be 2 or 3"
        ne = self.ne
        R = np.zeros((ne, dim * self.n))
        pts = self.pts if current else self.pts0
        

        for e in range(ne):
            i, j = self.edges[e]
            vec_ij = pts[j] - pts[i]
            dist_ij = np.linalg.norm(vec_ij)
            if dist_ij == 0: #avoid division by zero
                continue
            unit_ij = vec_ij / dist_ij
            for d in range(dim):
                R[e, dim * i + d] = -unit_ij[d]
                R[e, dim * j + d] = unit_ij[d]
        return R