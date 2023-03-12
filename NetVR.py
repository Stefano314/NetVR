import bpy

import numpy as np
import mathutils
from mathutils import Vector

from utils.canvas_manipulation import reset_canvas
from utils.files_reader import read_coordinates, read_edgelist


class Edge:
    """
    Description
    -----------
    Edge element class. This class creates an object that embeds all the properties of a link in a network.
    To be developed.
    """

    def __init__(self, extremes, weight=1., scale=0.01, name='link'):
        
        self.position = extremes
        self.weight = weight
        self.size = self.weight*scale
        self.name = name
        
        self.b_object = None
        
        self.attributes = None

class Node:
    """
    Description
    -----------
    Node element class. This class creates an object that embeds all the properties of a node in a network.
    To be developed.
    """

    def __init__(self, pos, size=1.0, name='node', color = [1.,1.,1.,1.]):

        self.position = pos
        self.size = size
        self.name = name
        
        self.b_object = None
        
        self.color = tuple(color)
        self.attributes = None
        
        

class Network:
    
    def __init__(self, nodes_coords, edgelist, radii = 1.0, weights = 1.0,
                 n_colors = None, e_colors = None):
        self.n_pos = nodes_coords - np.mean(nodes_coords, axis=0)
        self.e_list = edgelist
        
        self.e_pos = None
        self._get_e_pos()
        
        if isinstance(radii,list):
            pass
        else:
            radii = [radii]*len(self.n_pos)
        
        if isinstance(weights,list):
            pass
        else:
            weights = [weights]*len(self.e_pos)
            
        if n_colors is None:
            n_colors = np.ones((len(self.n_pos),4))
        else:
            pass
            
        self.edges = [Edge(e, w) for e, w in zip(self.e_pos, weights)]
        self.nodes = [Node(pos=n, size=r, color=c) for n, r, c in zip(self.n_pos, radii, n_colors)]
        
#        self.assign_textures(path = "C:/Users/stefa/Desktop/brain_graph.png")
    
        self.new = True #unused
    
    def assign_textures(self, path : str, mat_name = 'test_mat', tex_name = 'test_tex'):
        # Create a new material
        self.mats = bpy.data.materials.new(name=mat_name)
        self.mats.use_nodes = True
        
#        # Create a new image texture
#        self.tex = bpy.data.textures.new(name=tex_name, type='IMAGE')
#        image = bpy.data.images.load(path)
#        self.tex.image = image

#        # Create a texture slot in the material
#        slot = self.mats.texture_slots.add()
#        slot.texture = tex
#        slot.texture_coords = 'UV'
        

    def _get_e_pos(self):
        """
        Description
        -----------
        Evaluates the position of the center of the edges. This is needed when using the rotation of the links,
        since the rotation has the link center as rotation axis (by default, can be changed).
        """

        self.e_pos = np.zeros((len(self.e_list),3))
        
        for ind,e in enumerate(self.e_list):
            
            self.e_pos[ind] = ( (self.n_pos[e[0]][0]+self.n_pos[e[1]][0])/2.,\
                                (self.n_pos[e[0]][1]+self.n_pos[e[1]][1])/2.,\
                                (self.n_pos[e[0]][2]+self.n_pos[e[1]][2])/2. )
    
    # def _generate_edge_objects(self):
        
    #     pass
                                
    # def _generate_node_objects(self):
    #     # Check if the nodes sizes are given
    #     if isinstance(radius, list):
    #         pass
    #     else:
    #         radius = [radius]*len(self.nodes)
       
    def translate_to_point(self, point, draw=True):
        """
        Description
        -----------
        Translate and redraw the whole network in a specific point. The point will be the new network center.
        """

        point = np.array(point)
        vec = point - np.mean(self.n_pos, axis=0)
        self.n_pos += vec
        self.e_pos += vec
        
        for ind,e in enumerate(self.edges):
            e.position = self.e_pos[ind]
            
        for ind,n in enumerate(self.nodes):
            n.position = self.n_pos[ind]
            
        if draw:
            reset_canvas(all=False)
            self.draw_network()
    
    
    def scale_network(self, factor=2.):
        """
        Description
        -----------
        Scale network by a specified factor.
        """
        center = np.mean(self.n_pos, axis=0)
        self.translate_to_point([0.,0.,0.], draw=False)
        
        self.n_pos *= factor
        self.e_pos *= factor
        self.n_pos += center
        self.e_pos += center

        for ind,e in enumerate(self.edges):
            e.position = self.e_pos[ind]
            e.size *= factor
            
        for ind,n in enumerate(self.nodes):
            n.position = self.n_pos[ind]
            n.size *= factor
            
        reset_canvas(all=False)
        self.draw_network()
        
    
    def draw_nodes(self):

        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_uv_sphere_add(segments=18, ring_count=16, radius=1.)
        primitive_node = bpy.context.object
        
        for ind, center in enumerate(self.n_pos):
            
            node_size = self.nodes[ind].size
            
            # Create a new link_ object and link it to the mesh data block
            node_ = primitive_node.copy()
            node_.data = primitive_node.data.copy()
            
            node_.location = (self.n_pos[ind,0],self.n_pos[ind,1],self.n_pos[ind,2])
            node_.scale = (node_size, node_size, node_size)
            node_.name = f'node_{ind}'
            
            # Color
           
            mat = bpy.data.materials.new(name=f'node_{ind}')
            mat.diffuse_color = self.nodes[ind].color
            # Assign the material to the object
            if node_.data.materials:
                node_.data.materials[0] = mat
            else:
                node_.data.materials.append(mat)

            # Memorize object properties
            self.nodes[ind].name = f'node_{ind}'
            self.nodes[ind].b_object = node_
            
            bpy.context.scene.collection.objects.link(node_)
            
        # Remove primitive object
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects.get('Sphere').select_set(True)
        bpy.ops.object.delete()
        
#            bpy.context.view_layer.objects.active = node_

#            node_.active_material.diffuse_color = self.nodes[ind].color
            
#            bpy.data.materials.new(name=f"mat_node_{ind}")
#            bpy.data.materials[f"mat_node_{ind}"].diffuse_color = self.nodes[ind].color
#            bpy.data.materials[f"mat_node_{ind}"].specular_intensity = 0.5
#            mat.use_nodes = True
#            mat_nodes = mat.node_tree.nodes
#            mat_links = mat.node_tree.links
#            node_.data.materials.colors
#            mat_nodes['Principled BSDF'].inputs['Base Color'].default_value = self.nodes[ind].color
            
            
    
    def draw_edges(self):
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=1., radius2=1., depth=1.)
        primitive_link = bpy.context.object

        # Create links
        for ind,e in enumerate(self.e_list):
            
            e_weight = self.edges[ind].size
            v = self.n_pos[e[0]] - self.n_pos[e[1]]
            link_direction = Vector(v)
            link_length = link_direction.length
            link_direction.normalize()
            
            # Create a new link_ object and link it to the mesh data block
            link_ = primitive_link.copy()
            link_.data = primitive_link.data.copy()
            
            link_.location = self.e_pos[ind]
            
            link_.scale = (e_weight, e_weight, link_length)
            link_.name = f'link_{ind}'
            
            bpy.context.scene.collection.objects.link(link_)
            
            bpy.context.view_layer.objects.active = link_

            bpy.context.object.rotation_mode = 'QUATERNION'
            bpy.context.object.rotation_quaternion = link_direction.to_track_quat('Z','Y')
            
            # Memorize object properties
            self.edges[ind].name = f'link_{ind}'
            self.edges[ind].b_object = link_
            
        # Remove primitive object
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects.get('Cone').select_set(True)
        bpy.ops.object.delete()


    def draw_network(self):
        self.draw_nodes()
        self.draw_edges()
