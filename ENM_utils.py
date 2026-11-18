import plotly.graph_objects as go
from packing_utils import *
import numpy as np

class ENM:
    def __init__(self,net,k=None,l0=None):
        self.dim=net.dim
        self.n=net.n
        self.pts=net.pts
        self.pts0=net.pts.copy()
        self.graph=net.graph
        self.ne=len(net.graph.edges)
        self.edges=[(i,j) for i,j in net.graph.edges]
        
        if k is None:
            self.k = np.ones(self.ne)
        else:
            self.k = np.asarray(k)
            assert self.k.size == self.ne

        if l0 is None:
            l0 = np.zeros(self.ne)
            for ide,(i,j) in enumerate(self.edges):
                l0[ide]=np.linalg.norm(net.pts[i]-net.pts[j])
            self.l0=l0
        else:
            self.l0 = np.asarray(l0)
            assert self.l0.size == self.ne


    def plot_interactive(self, edges_a=None, nodes_a=None, nodes_b=None):
        def edge_mask(pairs):
            """Boolean mask over self.edges for the given node pairs (order-independent)."""
            want = {frozenset(p) for p in pairs}
            mask = np.array([frozenset(e) in want for e in self.edges], dtype=bool)
    
            found = {frozenset(e) for e in self.edges} & want
            if len(found) < len(want):
                missing = [sorted(p) for p in want - found]
                print(f"warning: {len(missing)} pair(s) are not edges in the network, e.g. {missing[:5]}")
            return mask
                
        def to_segs(ee):
            if len(ee) == 0:
                return np.empty((0, 3))
            nan = np.full((len(ee), 1, 3), np.nan)
            return np.concatenate([pts[ee], nan], axis=1).reshape(-1, 3)
        def nodes(idx, name, color, size):
            p = pts[idx]
            return go.Scatter3d(x=p[:,0], y=p[:,1], z=p[:,2], mode='markers', name=name,
                            marker=dict(size=size, color=color),
                            text=[str(i) for i in idx],
                            hovertemplate='node %{text}<extra></extra>')
        
        
        
        pts = np.asarray(self.pts); e = np.asarray(self.edges)
        mask = edge_mask(edges_a) if edges_a is not None else np.zeros(self.ne, bool)

        a = np.asarray(nodes_a if nodes_a is not None else [], int)
        b = np.asarray(nodes_b if nodes_b is not None else [], int)
        rest = np.setdiff1d(np.arange(self.n), np.concatenate([a, b]))

    

        bg, hi = to_segs(e[~mask]), to_segs(e[mask])
        fig = go.Figure([
            go.Scatter3d(x=bg[:,0], y=bg[:,1], z=bg[:,2], mode='lines',
                     line=dict(color='lightgray', width=2), name='bonds', hoverinfo='skip'),
            go.Scatter3d(x=hi[:,0], y=hi[:,1], z=hi[:,2], mode='lines',
                     line=dict(color='crimson', width=6), name='selected', hoverinfo='skip'),
            nodes(rest, 'nodes', 'dimgray', 3),
            nodes(a, 'type A', 'crimson', 7),
            nodes(b, 'type B', 'royalblue', 7),
            ])
        fig.update_layout(scene_aspectmode='data')
        return fig