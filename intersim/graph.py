# graph.py
from typing import List
import torch
class InteractionGraph:
    
    def __init__(self, neighbor_dict: dict={}):
        self._neighbor_dict = neighbor_dict
    
    def update_graph(self, x):
        """
        Update the neighbor dict given a new state.
        
        Args:
            x (torch.tensor): (nv,5) vehicle states
        """
        pass
    
    @classmethod
    def from_edges(cls, nodes: List[int], edges: List[tuple], *args, **kwargs):
        """
        Instantiate interaction graph from number of nodes and edges

        Args:
            nodes (list[int]): list of active node indices
            edges (list[tuple]): list of directed edges in graph
        """
        neighbor_dict = {i:[] for i in nodes} 
        for (i,j) in edges:
                assert i in nodes, "node {} from edge {} not present in active nodes list".format(i, (i,j))
                assert j in nodes, "node {} from edge {} not present in active nodes list".format(j, (i,j))
                if j not in neighbor_dict[i]:
                    neighbor_dict[i].append(j)
        return cls(*args, neighbor_dict=neighbor_dict, **kwargs)

    @property
    def neighbor_dict(self):
        return self._neighbor_dict.copy()
    
    @property
    def nodes(self):
        return sorted(list(self._neighbor_dict.keys()))
    
    @property
    def edges(self):
        return [(i,j) for i in self.nodes for j in self._neighbor_dict[i]]
    
    @property
    def reverse_edges(self):
        return [(j,i) for i in self.nodes for j in self._neighbor_dict[i]]
    
    def adjacency_matrix(self, nv):
        """
        Return adjacency matrix
        Args:
            nv (int): total number of vehicles in scene
        Returns:
            adj (torch.Tensor): (nv, nv) boolean tensor where adj[i,j] = (i,j) in edges
        """
        adj = torch.zeros(nv, nv, dtype=torch.bool)
        for (i,j) in self.edges:
            adj[i,j] = True
        return adj

    @property
    def reverse_neighbor_dict(self):
        rdict = {node:[] for node in self.nodes}
        for edge in self.reverse_edges:
            rdict[edge[0]].append(edge[1]) 
        return rdict
    
    @property
    def strongly_connected_components(self):
        """
        Compute the strongly connected components of the graph.
        Args:
        Returns:
            sccs (list of tuples): first element is vertices in each scc, second is outgoing edges
        """
        start = {n:-1 for n in self.nodes} # forward dfs start times
        fin = {n:-1 for n in self.nodes} # forward dfs end times
        self._visit_time = 0 
        for node in self.nodes: 
            if start[node] == -1: # if unvisited, recurse on subtree
                self.DFS(node, self._neighbor_dict, start, fin, [])
                
        # compute reverse neighbor dict
        rdict = self.reverse_neighbor_dict
        
        # compute node ordering based on reverse finish time
        sorted_items = sorted(fin.items(), key=lambda item:-item[1])
        reverse_ordering = [item[0] for item in sorted_items]
        
        start = {n:-1 for n in self.nodes} # backward dfs start times
        fin = {n:-1 for n in self.nodes} # backward dfs end times
        self._visit_time = 0 
        scc_list = []
        for node in reverse_ordering:
            if start[node] == -1: # new scc
                subtree = [node]
                self.DFS(node, rdict, start, fin, subtree)
                scc_list.append(sorted(subtree))
        
        # find outgoing edges
        sccs = []
        for scc in scc_list:
            outgoing_edges = []
            for node in scc:
                for neighbor_node in self._neighbor_dict[node]:
                    if neighbor_node not in scc and neighbor_node not in outgoing_edges:
                        outgoing_edges.append(neighbor_node)
            sccs.append((scc,sorted(outgoing_edges)))       
        return sccs
            
        
    def DFS(self, node: int, neighbor_dict: dict, start: dict, fin: dict, tree: list):
        """
        Perform a depth-first search of a subtree.
        Args:
            node (int): current root node of search
            neighbor_dict (dict): neighbor dictionary to use for traversal
            start (dict): dict of search start times
            fin (dict): dict of search end times
            tree (list): list of nodes in current tree
        Returns:
        """    

        self._visit_time += 1
        start[node] = self._visit_time
        for neighbor in neighbor_dict[node]:
            if start[neighbor] == -1: # recurse if unexplored
                tree.append(neighbor) # add neighbor to current subtree
                self.DFS(neighbor, neighbor_dict, start, fin, tree)
        fin[node] = self._visit_time
