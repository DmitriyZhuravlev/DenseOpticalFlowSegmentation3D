import numpy as np
import cv2
from collections import defaultdict, deque
from draw import generate_random_color
from lifting_3d import get_bottom_variants

class Node:
    def __init__(self, id, flow_value):
        """
        Initialize a Node.

        Args:
            id (int): The identifier for the node.
            flow_value: The flow value associated with the node.
        """
        self.id = id
        self.parent = id
        self.rank = 0
        self.size = 1
        self.flow_value = flow_value

    def __repr__(self):
        return '(parent=%s, rank=%s, size=%s)' % (self.parent, self.rank, self.size)

class Forest:
    def __init__(self, flow, bev, persp_mat, inv_mat, inv_mat_upper, min_move=5):
        """
        Initialize a Forest.

        Args:
            flow: Flow data.
            bev: Bird's-eye view data.
            persp_mat: Perspective matrix.
            inv_mat: Inverse matrix.
            inv_mat_upper: Upper inverse matrix.
            min_move (int): Minimum move threshold.
        """
        height, width, _ = flow.shape
        num_nodes = height * width
        self.nodes = [Node(i, flow[i // width, i % width]) for i in range(num_nodes)]
        self.num_sets = num_nodes
        self.segments = [set([i]) for i in range(num_nodes)]
        self.width = width
        self.height = height
        self.segment_scores = defaultdict(float)
        self.segment_history = [deque() for _ in range(num_nodes)]
        self.min_move = min_move
        self.bev = bev
        self.persp_mat = persp_mat
        self.inv_mat = inv_mat
        self.inv_mat_upper = inv_mat_upper

    def find(self, n):
        if n != self.nodes[n].parent:
            self.nodes[n].parent = self.find(self.nodes[n].parent)
        return self.nodes[n].parent

    def merge(self, a, b):
        parent_a = self.find(a)
        parent_b = self.find(b)

        if parent_a != parent_b:
            if self.nodes[parent_a].rank > self.nodes[parent_b].rank:
                parent_a, parent_b = parent_b, parent_a

            self.nodes[parent_a].parent = parent_b

            size_a = self.nodes[parent_a].size
            size_b = self.nodes[parent_b].size
            weighted_flow_a = np.multiply(self.nodes[parent_a].flow_value, size_a)
            weighted_flow_b = np.multiply(self.nodes[parent_b].flow_value, size_b)
            avg_flow = np.divide(np.add(weighted_flow_a, weighted_flow_b), size_a + size_b)

            self.nodes[parent_b].flow_value = avg_flow

            self.segments[parent_b].update(self.segments[parent_a])
            self.segments[parent_a].clear()
            self.nodes[parent_b].size += size_a
            self.nodes[parent_a].size = 0

            if self.nodes[parent_a].rank == self.nodes[parent_b].rank:
                self.nodes[parent_b].rank += 1

            self.num_sets -= 1

        return parent_b

    def new_merge(self, a, b, score_threshold=0.5, min_size=1000, min_move=1.0, min_convexity=1/2):
        parent_b = self.merge(a, b)

        if self.nodes[parent_b].size < min_size:
            return
        x, y = parent_b % self.width, parent_b // self.width
        if y < self.height/10:
            return

        move = np.linalg.norm(self.nodes[parent_b].flow_value)

        if move < (y + 1)/ self.height:
            return

        min_x, min_y, max_x, max_y = self.get_bounding_box(parent_b)
        convexity = self.nodes[parent_b].size / ((max_x - min_x + 1) * (max_y - min_y + 1))

        bbox = (min_x, min_y, max_x, max_y)

        score, solution = get_score(bbox, self.nodes[parent_b].flow_value, self.bev, self.persp_mat,
                                     self.inv_mat, self.inv_mat_upper)
        self.segment_scores[parent_b] = score

        if solution.cls == 0:
            min_convexity = 3/4
        if solution.cls == 1:
            min_convexity = 1/2
        if solution.cls == 2:
            min_convexity = 20/29

        if convexity < min_convexity:
            return

        if score > score_threshold:
            seg = self.segments[parent_b]
            self.segment_history[parent_b].append((score, list(seg), solution, move))

    def get_segment_best_score(self, node_id):
        return self.segment_scores[node_id]

    def get_best_segments(self):
        def key_function(item):
            score, segment, solution, move = item
            cls = -1 if solution is None else solution.cls
            return (score, len(segment))

        best_segments = {}

        for segment_id, history in enumerate(self.segment_history):
            if not history:
                continue

            best_score, best_segment, best_solution, best_move = max(history, key=key_function)

            if segment_id not in best_segments or best_score > best_segments[segment_id]["score"]:
                best_segments[segment_id] = {
                    "segment": best_segment,
                    "score": best_score,
                    "solution": best_solution,
                    "move": best_move
                }

        return list(best_segments.values())

    def get_segments(self):
        segments = {}
        for i in range(len(self.nodes)):
            root = self.find(i)
            if root not in segments:
                segments[root] = []
            segments[root].append(i)
        return list(segments.values())

    def get_bounding_box(self, node_id):
        segment = self.segments[self.find(node_id)]
        x_values = [node_id % self.width for node_id in segment]
        y_values = [node_id // self.width for node_id in segment]
        min_x = min(x_values)
        max_x = max(x_values)
        min_y = min(y_values)
        max_y = max(y_values)
        return min_x, min_y, max_x, max_y

def create_edge(flow, width, x, y, x1, y1, diff):
    vertex_id = lambda x, y: y * width + x
    w = diff(flow, x, y, x1, y1)
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

class Solution:
    def __init__(self, cls, ps_bev, lower_face, upper_face, rectangle, w_error, h_error, orient):
        self.cls = cls
        self.ps_bev = ps_bev
        self.lower_face = lower_face
        self.upper_face = upper_face
        self.rectangle = rectangle
        self.w_error = w_error
        self.h_error = h_error
        self.orient = orient

def get_score(bbox, direction, bev, persp_mat, inv_mat, inv_mat_upper):
    max_score = -1
    sol = None

    for cls in range(3):
        ps_bev, lower_face, upper_face, rectangle, w_error, h_error, deg = get_bottom_variants(direction, bbox,
                                                                                              persp_mat, inv_mat,
                                                                                              inv_mat_upper[cls], cls)
        if rectangle is not None and max_score < (w_error + h_error) / 2:
            max_score = (w_error + h_error) / 2
            sol = Solution(cls, ps_bev, lower_face, upper_face, rectangle, w_error, h_error, deg)

    return max_score, sol

def segment_graph(flow, graph_edges, bev, persp_mat, inv_mat, inv_mat_upper):
    forest = Forest(flow, bev, persp_mat, inv_mat, inv_mat_upper)
    weight = lambda edge: edge[2]
    sorted_graph = sorted(graph_edges, key=weight)

    for edge in sorted_graph:
        a = forest.find(edge[0])
        b = forest.find(edge[1])

        if a == b:
            continue

        forest.new_merge(a, b)

    return forest, sorted_graph
