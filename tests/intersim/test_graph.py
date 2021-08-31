# test_graph.py
from intersim.graph import InteractionGraph
from intersim.graphs import ConeVisibilityGraph, ClosestObstacleGraph
import torch
import numpy as np
torch.manual_seed(0)
new_state = (2*torch.rand(10,5) - 1) * torch.tensor([[1,1,1,np.pi,np.pi]])

def suite(graph):
    graph.strongly_connected_components
    graph.adjacency_matrix(10)
    graph.neighbor_dict
    graph.nodes
    graph.edges
    graph.update_graph(new_state)

def test_graph1():
    neighbor_dict = {0:[], 1:[2], 2:[]}
    graph = InteractionGraph(neighbor_dict)
    suite(graph)

def test_graph2():
    neighbor_dict = {0:[1], 1:[2], 2:[0]}
    graph = InteractionGraph(neighbor_dict)
    suite(graph)
    
def test_graph3():
    neighbor_dict = {0:[1], 1:[2,3], 2:[0], 3:[]}
    graph = InteractionGraph(neighbor_dict)
    suite(graph)

def test_graph4():
    neighbor_dict = {0:[1], 1:[0,2], 2:[3], 3:[2]}
    graph = InteractionGraph(neighbor_dict)
    suite(graph)
 
def test_graph5():
    neighbor_dict = {0:[1,2], 1:[0,5], 2:[0,5], 5:[6,8], 6:[8], 8:[5]}
    graph = InteractionGraph(neighbor_dict)
    suite(graph)

def test_graph6():
    neighbor_dict = {0:[1,2], 1:[0,5], 2:[0,5], 5:[6,8], 6:[8], 8:[5]}
    graph = ConeVisibilityGraph(neighbor_dict=neighbor_dict)
    suite(graph)
    
def test_graph7():
    neighbor_dict = {0:[1,2], 1:[0,5], 2:[0,5], 5:[6,8], 6:[8], 8:[5]}
    graph = ClosestObstacleGraph(neighbor_dict=neighbor_dict)
    suite(graph)