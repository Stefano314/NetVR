import numpy as np
import networkx as nx
import os
import trimesh

class NetVR:

    def __init__(self, nx_graph):
        
        self.G = nx_graph
        self.nodes = nx_graph.nodes(data=True)
        self.edges = nx_graph.edges(data=True)
        self.mesh = None
        self.centers = None
    
    def _set_nodes_position(self, N, r, scale=50):
        """
        Description
        -----------
        Generate N points in space that are at least distant 2*r with each other. This is just a way
        to create N non-overlapping spheres in a 3D space.

        Parameters
        ----------
        N : int
            Number of spheres to be represented.
        r : float, np.array
            Radius of each sphere. If a list is given, then the distances will vary accordingly.
        scale : int
            Scale factor that expands the dimension of the 3D space according to the maximum
            radius value.

        Returns
        -------
        points : numpy.array
            Points coordinates array. Each row is a point, the columns are in order (x,y,z). 
        """

        points = np.zeros((N, 3)) # Coordinates initialization
        box = 0 # 3D space dimension
        
        # Check if r is already a list
        if isinstance(r, np.ndarray):
            box=np.max(r)*scale
        else:
            box = r*scale
            r = [r]*N

        i = 0
        while i < N:
            # Generate random point in the box
            point = box*np.random.rand(3)

            # Check if the distance between the point and the rest is
            if i == 0 or np.min(np.linalg.norm(points[:i,:] - point, axis=1)) >= (r[i]+np.max(r[:i])):

                points[i,:] = point
                i += 1

        return points

    def create_mesh_model(self, radius = 1.0, width = 0.05, path='network_model.obj', save=True, pos=None):
        '''
        Description
        -----------
        Generate a mesh of the network. This will create a 3D model of the network, each node and edge will be a group, while the
        entire network is the object.

        Parameters
        ----------
        radius : float, list, optional
            Specify the radius of the spheres corresponding to the nodes. If the a list of radii is given, the nodes will be
            resized accordingly.
        width : float, list, optional
            Specify the width of the edges. If the list of edges widths is given, all the links will be resized accordingly.
        path : str, optional
            Path where to save the .OBJ object.
        save : bool, optional
            Specify if export the .OBJ model.
        pos : any, optional
            Specify the nodes positions in space. If it is not given, the nodes will be placed randomly and won't overlapp
            with each other.
        '''

        if pos is None:
            self.centers = self._set_nodes_position(len(self.nodes), radius)

        else:
            self.centers = pos

        mapp = {key:{k:v for (k,v) in zip(['x_pos','y_pos','z_pos'],[x_pos,y_pos,z_pos])} for 
                (key,x_pos,y_pos,z_pos) in zip(self.G.nodes(), self.centers[:,0], self.centers[:,1], self.centers[:,2])}
        
        nx.set_node_attributes(self.G, mapp)
        del mapp

        with open(path, 'w') as f:

            ind_obj = 1 # Node number
            ind_edges = 1 # Edge number
            n_vertices = 0 # Number of vertices in the mesh

            f.write(f'o {path[:-4]}\n')
            for point in self.centers:

                # Create spheres as nodes
                sphere = trimesh.primitives.Sphere(radius=radius, center=[point[0],point[1],point[2]], subdivisions=0)
                
                f.write(f"g node {ind_obj}\n")

                for row in sphere.vertices:
                    f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")
                # self.centers.append(sphere.center)
                
                for row in sphere.faces:
                    f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")

                ind_obj += 1
                n_vertices += len(sphere.vertices)

            for e in self.G.edges():
                
                # Define edges as boxes
                line = trimesh.creation.box(extents=(0.1, 0.1, 1))

                # Box trasformation to the vertices relative to the centers of the connected nodes 
                line.vertices[0] = [self.centers[e[0]-1,0]+width, self.centers[e[0]-1,1]+width, self.centers[e[0]-1,2]]
                line.vertices[1] = [self.centers[e[1]-1,0]+width, self.centers[e[1]-1,1]+width, self.centers[e[1]-1,2]]
                line.vertices[2] = [self.centers[e[0]-1,0]-width, self.centers[e[0]-1,1]-width, self.centers[e[0]-1,2]]
                line.vertices[3] = [self.centers[e[1]-1,0]-width, self.centers[e[1]-1,1]-width, self.centers[e[1]-1,2]]
                line.vertices[4] = [self.centers[e[0]-1,0]+width, self.centers[e[0]-1,1]+width, self.centers[e[0]-1,2]]
                line.vertices[5] = [self.centers[e[1]-1,0]+width, self.centers[e[1]-1,1]+width, self.centers[e[1]-1,2]]
                line.vertices[6] = [self.centers[e[0]-1,0]-width, self.centers[e[0]-1,1]-width, self.centers[e[0]-1,2]]
                line.vertices[7] = [self.centers[e[1]-1,0]-width, self.centers[e[1]-1,1]-width, self.centers[e[1]-1,2]]
            
                f.write(f'g edge {ind_edges}\n')
                
                for row in line.vertices:
                    f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")

                for row in line.faces:
                    f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")

                n_vertices += len(line.vertices)
                ind_edges+=1

        # Create mesh object
        self.mesh = trimesh.load(path, force='mesh')

        # Delete file if not required
        if not save:
            os.remove(path)

    def export_to_obj(self, path : str, **kwargs):
        trimesh.exchange.export.export_mesh(self.mesh, path, kwargs, file_type='obj')
