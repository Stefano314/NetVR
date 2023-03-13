import numpy as np
import networkx as nx
import os
import trimesh

# Directions with all the diagonals
directions = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), 
              (-1, 0, 0), (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
              (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
              (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
              (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
              (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

def is_valid_index(index, A):
    return all(0 <= i < A.shape[0] for i in index)
    
def undirected_edges(edgelist):
    reversed_edges = {}
    undirected_edgelist = []

    for edge in edgelist:
        reversed_edge = (edge[1], edge[0])
        if reversed_edge in reversed_edges:
            continue
        else:
            reversed_edges[reversed_edge] = True
            undirected_edgelist.append(edge)

    return undirected_edgelist

def _set_nodes_position(N, r, scale=50):
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

def array_mask_mesh(A : np.ndarray, radius=1., width=0.05, edges = False, 
                    n_category=None, path : str='test.obj', volume='cube'):
    """
    Create a 3D object of a (N,M,L) array mask.
    If volume = 'cube', edges=False and radius=1, then the boxes will share faces, producing a real volume mask,
    but it's harder to understand visually.
    """
    # Create volumes even for 1D and 2D arrays
    if len(A.shape) == 2:
        new = np.zeros((A.shape[0],A.shape[1],2))
        new[:,:,0]=A
        A=new
        del new
        
    if len(A.shape) == 1:
        new = np.zeros((A.shape[0],2,2))
        new[:,0,0]=A
        A=new
        del new

    centers = np.argwhere(A)
    radius = [radius]*centers.shape[0]
    
    mesh = None

    # All elements are the same if not specified
    # !No edges category, since in this case the edges are surfaces!
    if n_category is None:
        n_category = [None]*len(centers)
    prev_cat = None

    with open(path, 'w') as f:
        
        f.write('# Generated with NetVR https://github.com/Stefano314/NetVR \n\n')
        
        ind_edges = 1 # Edge number
        n_vertices = 0 # Number of vertices in the mesh

        f.write(f'o {path[:-4]}\n') # Object name
        for point, r, ind_obj in zip(centers, radius, n_category):
            
            # Create spheres as nodes
            if volume == 'cube':
                # Construct the translation matrix
                translation_matrix = np.array([[1, 0, 0, point[0]], 
                                               [0, 1, 0, point[1]], 
                                               [0, 0, 1, point[2]], 
                                               [0, 0, 0, 1]])
                mesh = trimesh.creation.box(extents=(r, r, r), transform=translation_matrix)
            elif volume == 'sphere':
                mesh = trimesh.primitives.Sphere(radius=r, center=[point[0],point[1],point[2]], subdivisions=0)

            if ind_obj != prev_cat:# Write only new categories
                f.write(f"g node {ind_obj}\n")

            prev_cat = ind_obj

            for row in mesh.vertices:
                f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")
            
            for row in mesh.faces:
                f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")

            ind_obj += 1
            n_vertices += len(mesh.vertices)

        if edges:

            # Iterate over all coordinates and check adjacent elements
            edgelist = []
            # center_index = 0
            
            for index in centers:
                starting_index = np.where(np.all(centers==list(index), axis=1))[0][0]
                # for offset in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:# No diag
                for offset in directions:# With diag

                    adjacent_index = tuple(index + offset)
                    if is_valid_index(adjacent_index, A) and A[adjacent_index]:
                        edgelist.append((starting_index, np.where(np.all(centers==list(adjacent_index), axis=1))[0][0]))

            edgelist = undirected_edges(edgelist) # Keep it undirected (waaaay lighter obj)

            # edgelist = [(e[0]-1,e[1]-1) for e in edgelist] # In case we don't start from zero position
            for e in edgelist:
                
                line = trimesh.creation.cylinder(radius=width, segment=[centers[e[0]],centers[e[1]]], sections=4)

                if ind_edges == 1: # Create only one edge group for the moment
                    f.write(f'g edge {ind_edges}\n')
                
                for row in line.vertices:
                    f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")

                for row in line.faces:
                    f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")

                n_vertices += len(line.vertices)
                ind_edges+=1

def create_mesh_model(nodes, G, radius = 1.0, width = 0.05, path='network_model.obj', save=True, pos=None):
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
    # Check if the nodes sizes are given
    if isinstance(radius, list):
        pass
    else:
        radius = [radius]*len(nodes)
    # Check if the position is given
    if pos is None:
        centers = _set_nodes_position(len(nodes), radius)
    else:
        centers = pos
    # Add position attribute to network
    mapp = {key:{k:v for (k,v) in zip(['x_pos','y_pos','z_pos'],[x_pos,y_pos,z_pos])} for 
            (key,x_pos,y_pos,z_pos) in zip(G.nodes(), centers[:,0], centers[:,1], centers[:,2])}
    
    nx.set_node_attributes(G, mapp)
    del mapp # Free space
    # Create .OBJ file
    with open(path, 'w') as f:
        
        f.write('# Generated with NetVR https://github.com/Stefano314/NetVR \n\n')
        
        ind_obj = 1 # Node number
        ind_edges = 1 # Edge number
        n_vertices = 0 # Number of vertices in the mesh
        f.write(f'o {path[:-4]}\n') # Object name
        for point, r in zip(centers, radius):
            # Create spheres as nodes
            sphere = trimesh.primitives.Sphere(radius=r, center=[point[0],point[1],point[2]], subdivisions=3)
            
            f.write(f"g node {ind_obj}\n")
            for row in sphere.vertices:
                f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")
            # centers.append(sphere.center)
            
            for row in sphere.faces:
                f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")
            ind_obj += 1
            n_vertices += len(sphere.vertices)

        for e in G.edges():
            # -1 is for edges in networkx
            line = trimesh.creation.cylinder(radius=width, segment=[centers[e[0]-1],centers[e[1]-1]], sections=4)
        
            f.write(f'g edge {ind_edges}\n')
            
            for row in line.vertices:
                f.write(f"v {np.round(row[0],5)} {np.round(row[1],5)} {np.round(row[2],5)}\n")
            for row in line.faces:
                f.write(f"f {row[0]+n_vertices+1} {row[1]+n_vertices+1} {row[2]+n_vertices+1}\n")
            n_vertices += len(line.vertices)
            ind_edges+=1
    # # Create mesh object
    # mesh = trimesh.load(path, force='mesh')
    # Delete file if not required
    if not save:
        os.remove(path)

def read_coordinates(path : str, header = True):
    a = []
    coords = []
    categories = []

    with open(path, 'r') as f:
        a = [line.replace('\n','') for line in f]


    if len(a[0].split(',')) == 4:
        # csv: x,y,z,category
        if header:
            coords = np.array([[float(i.split(',')[0]),
                            float(i.split(',')[1]),
                            float(i.split(',')[2])] for i in a[1:]])
            
            categories = [str(i.split(',')[3]) for i in a[1:]]

        else:
            coords = np.array([[float(i.split(',')[0]),
                            float(i.split(',')[1]),
                            float(i.split(',')[2])] for i in a])
            
            categories = [str(i.split(',')[3]) for i in a]

    else:
        # csv: x,y,z coordinates
        categories = ['white']*len(a)

        if header:
            coords = np.array([[float(i.split(',')[0]),
                            float(i.split(',')[1]),
                            float(i.split(',')[2])] for i in a[1:]])
        else:
            coords = np.array([[float(i.split(',')[0]),
                            float(i.split(',')[1]),
                            float(i.split(',')[2])] for i in a])
    return coords, categories
        