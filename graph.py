import numpy as np


class Node:
    def __init__(self, parent, color=(0, 0, 0), box = [0, 0, 0, 0], rank=0, size=1):
        self.parent = parent
        self.rank = rank
        self.size = size
        self.color = np.array(color, dtype=np.float32)
        self.box = box


    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)

class Forest:
    def __init__(self, num_nodes, img = None, width = None):
        if img is not None:
            self.nodes = [Node(i, img[i % width][ i // width ], [i % width,  i // width, i % width,  i // width] ) for i in range(num_nodes)]
        else:
             self.nodes = [Node(i) for i in range(num_nodes)]
        self.num_sets = num_nodes

    def size_of(self, i):
        return self.nodes[i].size

    def find(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent

        self.nodes[n].parent = temp
        return temp
        
    def color(self, n):
        temp = n
        while temp != self.nodes[temp].parent:
            temp = self.nodes[temp].parent

        self.nodes[n].parent = temp
        return self.nodes[n].color


    def merge(self, a, b):
        if self.nodes[a].rank > self.nodes[b].rank:
            self.nodes[b].parent = a
            self.nodes[a].color = (self.nodes[a].size * self.nodes[a].color + self.nodes[b].size * self.nodes[b].color)/(self.nodes[b].size + self.nodes[a].size)
            self.nodes[a].size = self.nodes[a].size + self.nodes[b].size
            self.nodes[a].box = [min(self.nodes[b].box[0], self.nodes[a].box[0]), min(self.nodes[b].box[1], self.nodes[a].box[1]), max(self.nodes[b].box[0], self.nodes[a].box[0]), max(self.nodes[b].box[1], self.nodes[a].box[1])]
        else:
            self.nodes[a].parent = b
            self.nodes[b].color = (self.nodes[a].size * self.nodes[a].color + self.nodes[b].size * self.nodes[b].color)/(self.nodes[b].size + self.nodes[a].size)
            self.nodes[b].size = self.nodes[b].size + self.nodes[a].size
            self.nodes[b].box = [min(self.nodes[b].box[0], self.nodes[a].box[0]), min(self.nodes[b].box[1], self.nodes[a].box[1]), max(self.nodes[b].box[0], self.nodes[a].box[0]), max(self.nodes[b].box[1], self.nodes[a].box[1])]

            if self.nodes[a].rank == self.nodes[b].rank:
                self.nodes[b].rank = self.nodes[b].rank + 1

        self.num_sets = self.num_sets - 1

    def print_nodes(self):
        for node in self.nodes:
            print(node)

def create_edge(img, width, x, y, x1, y1, diff):
    vertex_id = lambda x, y: y * width + x
    w = diff(img, x, y, x1, y1)
    return (vertex_id(x, y), vertex_id(x1, y1), w)

def build_graph(img, width, height, diff, neighborhood_8=False):
    graph_edges = []
    for y in range(height):
        for x in range(width):
            if x > 0:
                graph_edges.append(create_edge(img, width, x, y, x-1, y, diff))

            if y > 0:
                graph_edges.append(create_edge(img, width, x, y, x, y-1, diff))

            if neighborhood_8:
                if x > 0 and y > 0:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y-1, diff))

                if x > 0 and y < height-1:
                    graph_edges.append(create_edge(img, width, x, y, x-1, y+1, diff))

    return graph_edges

def remove_small_components(forest, graph, min_size):
    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b and (forest.size_of(a) < min_size or forest.size_of(b) < min_size):
            forest.merge(a, b)

    return  forest
    
def merge_components(forest, graph, flow, diff, width, num_nodes, threshold_func, const):
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]

    for edge in graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a != b:
            # y = a // width
            # x = a % width
            # y1 = b // width
            # x1 = b % width
            #weight = diff(flow, x, y, x1, y1)
            out = np.sum((forest.color(a) - forest.color(b)) ** 2)
            weight = np.sqrt(out)
            print("weight :", weight)
            # a_condition = weight <= threshold[a]
            # b_condition = weight <= threshold[b]
            #if (weight + edge[2])/2 < 5: #a_condition and b_condition:
            if weight < 5 and edge[2] < 5:
                forest.merge(a, b)
                # a = forest.find(a)
                # threshold[a] = weight + threshold_func(forest.nodes[a].size, const)

    return  forest

def segment_graph(img, graph_edges, num_nodes, const, min_size, threshold_func, width):
    # Step 1: initialization
    #forest = Forest(num_nodes)
    forest = Forest(num_nodes, img, width)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key=weight)
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]

    # Step 2: merging
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)

    return remove_small_components(forest, sorted_graph, min_size)


def segment_graph_flow(flow, graph_edges, num_nodes, const,  min_size, threshold_func, diff, width):
    # Step 1: initialization
    forest = Forest(num_nodes, flow, width)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key=weight)
    threshold = [ threshold_func(1, const) for _ in range(num_nodes) ]

    # Step 2: merging
    for edge in sorted_graph:
        parent_a = forest.find(edge[0])
        parent_b = forest.find(edge[1])
        a_condition = weight(edge) <= threshold[parent_a]
        b_condition = weight(edge) <= threshold[parent_b]

        if parent_a != parent_b and a_condition and b_condition:
            forest.merge(parent_a, parent_b)
            a = forest.find(parent_a)
            threshold[a] = weight(edge) + threshold_func(forest.nodes[a].size, const)
    
    forest = remove_small_components(forest, sorted_graph, min_size)

    return merge_components(forest, sorted_graph, flow, diff, width, num_nodes, threshold_func, const)
