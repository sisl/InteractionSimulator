# test_scc.py
from intersim.graph import InteractionGraph
def main():
    
    neighbor_dict1 = {0:[], 1:[2], 2:[]}
    graph1 = InteractionGraph(neighbor_dict1)
    print(graph1.strongly_connected_components)

    neighbor_dict2 = {0:[1], 1:[2], 2:[0]}
    graph2 = InteractionGraph(neighbor_dict2)
    print(graph2.strongly_connected_components)
    
    neighbor_dict3 = {0:[1], 1:[2,3], 2:[0], 3:[]}
    graph3 = InteractionGraph(neighbor_dict3)
    print(graph3.strongly_connected_components)
    
    neighbor_dict4 = {0:[1], 1:[0,2], 2:[3], 3:[2]}
    graph4 = InteractionGraph(neighbor_dict4)
    print(graph4.strongly_connected_components)
 
    neighbor_dict5 = {0:[1,2], 1:[0,5], 2:[0,5], 5:[6,8], 6:[8], 8:[5]}
    graph5 = InteractionGraph(neighbor_dict5)
    print(graph5.strongly_connected_components)
    
if __name__ == '__main__':
    main()