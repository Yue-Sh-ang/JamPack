import numpy as np
from poisson_disc import Bridson_sampling
from scipy.spatial import Voronoi, Delaunay
import networkx as nx
from scipy.integrate import solve_ivp
from tqdm import tqdm
from numba import jit

class Packing:
	'''Class to model a contact network of bidisperse spheres.

	Parameters
	----------
	n : int
		The number of spheres.
	dim : int
		The dimensionality of the system. Valid options are 2 and 3.
	radius : float, optional
		The radius of the larger spheres. If 0 (default), a good choice is computed
		automatically.
	rfac : float, optional
		Factor by which the smaller sphere radius is scaled relative to large spheres.
	params : dict, optional
		Specifies system parameters. Required keywords are :

		- 'central': strength of the force pulling spheres toward origin
		- 'drag': coefficient of isotropic drag
		- 'contact': strength of contact repulsion

	seed : int, optional
		A random seed used for initializing the positions of the spheres, and assigning
		which are large or small.

	Attributes
	----------
	n : int
		The number of spheres.
	dim : int
		The dimensionality of the system.
	pts : ndarray
		(x,y) coordinates for each sphere.
	labels : ndarray
		Array of sphere labels (0=large, 1=small).
	radii : list
		List of sphere radii, indexed by sphere type (0=large, 1=small).
	traj : ndarray
		The simulated trajectory of the network produced after a call to the generate() routine.
	t_eval : ndarray
		The corresponding time at each simulated frame.
	graph : networkx.graph
		Graph specifying the nodes and edges in the final contact network.
	'''

	def __init__(self, n, dim=2, radius=0, rfac=0.8,
				 params={'central':0.0005, 'drag':0.05, 'contact':0.1}, seed=12):

		if (dim != 2) and (dim != 3):
			raise ValueError("Dimension must be 2 or 3.")

		self.n = n
		self.dim = dim
		self.radius = radius
		self.rfac = rfac
		self.params = params
		self.seed = seed
		self.graph = None
		self.initialize()

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											DISK INITIALIZATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def initialize(self):
		'''Initialize the sphere positions and radii.'''

		self._initialize_points()
		self._initialize_labels()
		self.radii = [self.radius, self.rfac*self.radius]

	def _distance(self, p, q):
		'''Compute the distance between two coordinate pairs p and q.

		Parameters
		----------
		p : ndarray
			The positions, or array of positions, of the first point.
		q : ndarray
			The positions, or array of positions, of the second point.

		Returns
		-------
		float or ndarray
			The distance between the two points.
		'''

		return np.sqrt(np.sum(np.square(p - q), axis=-1))

	def _set_radius(self):
		'''Automatically set the radius of the large spheres.'''

		sfac = 0.5 # safety factor.

		# Compute radius of n spheres that perfectly fill a sphere inscribed
		# in a unit cube, with safety factor.
		self.radius = sfac/2*pow(self.n, -1./self.dim)

	def _initialize_points(self):
		'''Randomly initialize spheres using Poisson disc sampling.'''

		np.random.seed(self.seed)
		if self.radius == 0:
			self._set_radius()

		pts = Bridson_sampling(dims=np.ones(self.dim), radius=2*self.radius)
		if self.dim == 2:
			pts = np.vstack([pts.T, np.zeros(len(pts))]).T # augment to 3D.

		if len(pts) < self.n:
			print("Found only {:d} points; retry with smaller separation distance.".format(len(pts)))
		else:
			com = np.mean(pts, axis=0)
			idx = np.argsort(self._distance(pts, com))
			pts = pts[idx[:self.n]]
			self.pts = pts - np.mean(pts, axis=0)

	def _initialize_labels(self):
		'''Label spheres as large (0) or small (1) at random.'''

		np.random.seed(self.seed)
		self.labels = np.random.randint(2, size=self.n)

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											GRAPH GENERATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def _generate_graph(self):
		'''Generate the sphere contact graph.'''

		dpos = {i:self.pts[i] for i in range(self.n)}
		# first, create graph with no edges. We will add in all the edges individually.
		self.graph = nx.random_geometric_graph(self.n, 0, seed=self.seed, pos=dpos)
		for a,b in zip([0,0,1],[0,1,1]):
			d = self.radii[a] + self.radii[b]
			g = nx.random_geometric_graph(self.n, d, seed=self.seed, pos=dpos)
			for edge in g.edges():
				i = edge[0]; j = edge[1]
				if (self.labels[i] == a and self.labels[j] == b) or \
					(self.labels[i] == b and self.labels[j] == a):
					self.graph.add_edge(i,j)

	def coordination(self):
		'''Compute the coordination information of the network.

		Returns
		-------
		nc : int
			The number of nodes forming a connected network.
		Ziso : float
			The required average coordination for isostaticity.
		Z : float
			The average coordination of the network.
		dZ : float
			The excess coordination, defined as Z - Ziso.
		'''

		if self.graph is None:
			raise ValueError('Generate must be called first to produce a graph.')

		degree = np.array(list(self.graph.degree[i] for i in range(self.n)), dtype=int)
		ne = len(self.graph.edges())
		nc = self.n-np.sum(degree<2).astype(int) # connected nodes (excluding dangling nodes)
		if nc > 0:
			Z = 2*ne/float(nc)
			Ziso = 2*self.dim - self.dim*(self.dim+1)/float(nc)
			dZ = Z - Ziso
		else:
			Ziso = Z = dZ = 0

		return nc, Ziso, Z, dZ

	def save(self, filename):
		'''Save the contact network to a file.

		Parameters
		----------
		filename : str
			The name of the text file to write.
		'''

		ne = len(self.graph.edges())
		with open(filename, 'w') as f:
			f.write(str(self.dim)+'\n')
			f.write(str(self.n)+'\n')
			for i in range(self.n):
				f.write('{:.15g} {:.15g} {:.15g}\n'.format(self.pts[i,0],self.pts[i,1],self.pts[i,2]))
			f.write(str(ne)+'\n')
			for edge in self.graph.edges():
				f.write('{:d} {:d}\n'.format(edge[0],edge[1]))

	'''
	*****************************************************************************************************
	*****************************************************************************************************

											PACKING SIMULATION

	*****************************************************************************************************
	*****************************************************************************************************
	'''

	def generate(self, duration=1000., frames=200):
		'''Run a packing simulation.

		Generates a graph of the resulting contact network at the end.

		Parameters
		----------
		duration : float, optional
			The final integration time.
		frames : int, optional
			The number of output frames to produce (excluding initial frame).
		'''

		self._generate_packing(duration, frames)
		self._generate_graph()


	def _generate_packing(self, duration, frames):
		'''Routine for integrating the ODE system of packing spheres.

		Parameters
		----------
		duration : float
			The final integration time.
		frames : int
			The number of output frames to produce (excluding initial frame).
		'''

		q = np.hstack([self.pts.ravel(),np.zeros(3*self.n)])
		ti = 0; tf = duration
		t_span = [ti, tf]
		self.t_eval = np.linspace(ti, tf, frames+1)

		self.tp = ti
		with tqdm(total=tf-ti, unit='sim. time', initial=ti, ascii=True,
				  bar_format='{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}]', desc='progress') as self.pbar:
			sol = solve_ivp(self._ff, t_span, q, t_eval=self.t_eval, jac=self._jj, method='LSODA')

		q = sol.y.T[:,:3*self.n]
		self.traj = q.reshape(frames+1, self.n, 3)
		self.pts = self.traj[-1]

	def _ff(self, t, q):
		'''ODE system for modeling packed spheres.

		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The sphere position and velocity degrees of freedom.
		'''

		n = self.n
		pos, vel = q[:3*n], q[3*n:]
		acc = np.zeros(3*n)

		# compute central force and drag.
		b = self.params['drag']
		c = self.params['central']
		self._central_force(t, n, pos, vel, acc, b, c)

		# compute contact forces.
		c = self.params['contact']
		dpos = {i:[pos[3*i],pos[3*i+1],pos[3*i+2]] for i in range(n)}
		for a, b in zip([0,0,1],[0,1,1]):
			d = self.radii[a] + self.radii[b]
			g = nx.random_geometric_graph(n, d, seed=self.seed, pos=dpos)
			ne = len(g.edges())
			edge_i = np.zeros(ne, dtype=int)
			edge_j = np.zeros(ne, dtype=int)
			pair_type = np.zeros(ne, dtype=bool) # to mark if these are the correct pair type.
			for e, edge in enumerate(g.edges()):
				i, j = edge[0], edge[1]
				edge_i[e] = i; edge_j[e] = j
				pair_type[e] = (self.labels[i] == a and self.labels[j] == b) or \
			   				   (self.labels[i] == b and self.labels[j] == a)
			self._contact_force(t, n, pos, acc, c, d, (edge_i, edge_j, pair_type))

		dq = np.hstack([vel, acc])

		# update progress bar
		dt = t - self.tp
		self.pbar.update(dt)
		self.tp = t

		return dq

	def _jj(self, t, q):
		'''The jacobian of the ODE system, required by implicit integrators.

		Parameters
		----------
		t : float
			The current time.
		q : ndarray
			The sphere position and velocity degrees of freedom.
		'''

		n = self.n
		pos, vel = q[:3*n], q[3*n:]
		jac = np.zeros((6*n, 6*n))

		# compute central force and drag jacobian.
		b = self.params['drag']
		c = self.params['central']
		self._central_jacobian(t, n, pos, vel, jac, b, c)

		# compute contact jacobian.
		c = self.params['contact']
		dpos = {i:[pos[3*i],pos[3*i+1],pos[3*i+2]] for i in range(n)}
		for a, b in zip([0,0,1],[0,1,1]):
			d = self.radii[a] + self.radii[b]
			g = nx.random_geometric_graph(n, d, seed=self.seed, pos=dpos)
			ne = len(g.edges())
			edge_i = np.zeros(ne, dtype=int)
			edge_j = np.zeros(ne, dtype=int)
			pair_type = np.zeros(ne, dtype=bool) # to mark if these are the correct pair type.
			for e, edge in enumerate(g.edges()):
				i, j = edge[0], edge[1]
				edge_i[e] = i; edge_j[e] = j
				pair_type[e] = (self.labels[i] == a and self.labels[j] == b) or \
			   				   (self.labels[i] == b and self.labels[j] == a)
			self._contact_jacobian(t, n, pos, jac, c, d, (edge_i, edge_j, pair_type))

		return jac

	@staticmethod
	@jit(nopython=True)
	def _central_force(t, n, pos, vel, acc, b, c):
		'''Apply a central force pulling spheres toward origin.

		Also stabilizes the simulation with isotropic drag.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of spheres.
		pos : ndarray
			The sphere positions.
		vel : ndarray
			The sphere velocities.
		acc : ndarray
			The array in which to populate sphere accelerations on output.
		b : float
			Drag coefficient.
		c : float
			Central force strength.
		'''

		for i in range(n):
			x, y, z = pos[3*i], pos[3*i+1], pos[3*i+2]
			vx, vy, vz = vel[3*i], vel[3*i+1], vel[3*i+2]
			acc[3*i] -= b*vx + c*x
			acc[3*i+1] -= b*vy + c*y
			acc[3*i+2] -= b*vz + c*z

	@staticmethod
	@jit(nopython=True)
	def _central_jacobian(t, n, pos, vel, jac, b, c):
		'''Compute the jacobian of the central forces.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of spheres.
		pos : ndarray
			The sphere positions.
		vel : ndarray
			The sphere velocities.
		jac : ndarray
			The array in which to populate jacobian entries on output.
		b : float
			Drag coefficient.
		c : float
			Central force strength.
		'''

		for i in range(n):
			x, y, z = pos[3*i], pos[3*i+1], pos[3*i+2]
			vx, vy, vz = vel[3*i], vel[3*i+1], vel[3*i+2]

			jac[3*i,3*n+3*i] += 1.
			jac[3*i+1,3*n+3*i+1] += 1.
			jac[3*i+2,3*n+3*i+2] += 1.

			jac[3*n+3*i,3*i] -= c
			jac[3*n+3*i+1,3*i+1] -= c
			jac[3*n+3*i+2,3*i+2] -= c


			jac[3*n+3*i,3*n+3*i] -= b
			jac[3*n+3*i+1,3*n+3*i+1] -= b
			jac[3*n+3*i+2,3*n+3*i+2] -= b

	@staticmethod
	@jit(nopython=True)
	def _contact_force(t, n, pos, acc, c, d, edges):
		'''Apply a repulsive force between spheres in contact.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of spheres.
		pos : ndarray
			The sphere positions.
		acc : ndarray
			The array in which to populate sphere accelerations on output.
		c : float
			The contact interaction strength.
		d : float
			The equilibrium separation.
		edges : tuple
			Network edges.
		'''

		edge_i, edge_j, pair_type = edges
		for e, (i, j) in enumerate(zip(edge_i,edge_j)):
			if pair_type[e]:
				xi, yi, zi = pos[3*i], pos[3*i+1], pos[3*i+2]
				xj, yj, zj = pos[3*j], pos[3*j+1], pos[3*j+2]
				dx = xi-xj; dy = yi-yj; dz = zi-zj
				r = np.sqrt(dx**2 + dy**2 + dz**2)
				fac = -c*(1 - d/r)
				fx = fac*dx
				fy = fac*dy
				fz = fac*dz
				acc[3*i] += fx; acc[3*i+1] += fy; acc[3*i+2] += fz
				acc[3*j] -= fx; acc[3*j+1] -= fy; acc[3*j+2] -= fz

	@staticmethod
	@jit(nopython=True)
	def _contact_jacobian(t, n, pos, jac, c, d, edges):
		'''Compute the jacobian of the repulsive forces.

		Parameters
		----------
		t : float
			The current time.
		n : int
			The number of spheres.
		pos : ndarray
			The sphere positions.
		jac : ndarray
			The array in which to populate jacobian entries on output.
		c : float
			The contact interaction strength.
		d : float
			The equilibrium separation.
		edges : tuple
			Network edges.
		'''

		edge_i, edge_j, pair_type = edges
		for e, (i, j) in enumerate(zip(edge_i,edge_j)):
			if pair_type[e]:
				xi, yi, zi = pos[3*i], pos[3*i+1], pos[3*i+2]
				xj, yj, zj = pos[3*j], pos[3*j+1], pos[3*j+2]
				dx = xi-xj; dy = yi-yj; dz = zi-zj
				r2 = dx**2 + dy**2 + dz**2
				r = np.sqrt(r2); r3 = r2*r
				xx = -c*(d/r*(dx*dx/r2-1)+1)
				yy = -c*(d/r*(dy*dy/r2-1)+1)
				zz = -c*(d/r*(dz*dz/r2-1)+1)
				xy = -c*d*dx*dy/r3
				xz = -c*d*dx*dz/r3
				yz = -c*d*dy*dz/r3

				jac[3*n+3*i,3*i] += xx # xixi
				jac[3*n+3*i+1,3*i+1] += yy # yiyi
				jac[3*n+3*i+2,3*i+2] += zz # zizi
				jac[3*n+3*i,3*i+1] += xy # xiyi
				jac[3*n+3*i+1,3*i] += xy # yixi
				jac[3*n+3*i,3*i+2] += xz # xizi
				jac[3*n+3*i+2,3*i] += xz # zixi
				jac[3*n+3*i+1,3*i+2] += yz # yizi
				jac[3*n+3*i+2,3*i+1] += yz # ziyi

				jac[3*n+3*j,3*j] += xx # xjxj
				jac[3*n+3*j+1,3*j+1] += yy # yjyj
				jac[3*n+3*j+2,3*j+2] += zz # zjzj
				jac[3*n+3*j,3*j+1] += xy # xjyj
				jac[3*n+3*j+1,3*j] += xy # yjxj
				jac[3*n+3*j,3*j+2] += xz # xjzj
				jac[3*n+3*j+2,3*j] += xz # zjxj
				jac[3*n+3*j+1,3*j+2] += yz # yjzj
				jac[3*n+3*j+2,3*j+1] += yz # zjyj

				jac[3*n+3*i,3*j] -= xx # xixj
				jac[3*n+3*j,3*i] -= xx # xjxi
				jac[3*n+3*i+1,3*j+1] -= yy # yiyj
				jac[3*n+3*j+1,3*i+1] -= yy # yjyi
				jac[3*n+3*i+2,3*j+2] -= zz # zizj
				jac[3*n+3*j+2,3*i+2] -= zz # zjzi

				jac[3*n+3*i,3*j+1] -= xy # xiyj
				jac[3*n+3*j,3*i+1] -= xy # xjyi
				jac[3*n+3*i+1,3*j] -= xy # yixj
				jac[3*n+3*j+1,3*i] -= xy # yjxi

				jac[3*n+3*i,3*j+2] -= xz # xizj
				jac[3*n+3*j,3*i+2] -= xz # xjzi
				jac[3*n+3*i+2,3*j] -= xz # zixj
				jac[3*n+3*j+2,3*i] -= xz # zjxi

				jac[3*n+3*i+1,3*j+2] -= yz # yizj
				jac[3*n+3*j+1,3*i+2] -= yz # yjzi
				jac[3*n+3*i+2,3*j+1] -= yz # ziyj
				jac[3*n+3*j+2,3*i+1] -= yz # zjyi


