import numpy as np
import networkx as nx
import os
import trimesh

class NetVR:

    def __init__(self, nx_graph):
        self.G = nx_graph
        # self.n_objects = len(nx_obj.nodes()) useless?
        self.nodes = nx_graph.nodes(data=True)
        self.edges = nx_graph.edges(data=True)
        # self.centers = []
        self.mesh = None
    
    def _set_nodes_position(self, N, r, scale=50):
        points = np.zeros((N, 3))
        i = 0
        box=r*scale # The box is 10 times the sphere radius
        
        while i < N:
            point = box*np.random.rand(3)

            if i == 0 or np.min(np.linalg.norm(points[:i,:] - point, axis=1)) >= 2*r:
                points[i,:] = point
                i += 1

        return points

    def create_mesh_model(self, radius = 1.0, width = 0.05, path='network_model.obj', save=True, pos=None):

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
