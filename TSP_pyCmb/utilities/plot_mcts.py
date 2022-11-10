import networkx as nx
from matplotlib import pyplot as plt


def plot_mcts(network, cost, runtime, edges):
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        "Path Found by Different Models (num_of_node={:d}, side_length={:d})".format(num_of_node, side_length))
    model_names = ['greedy heuristic', '2-opt heuristic', 'random mcts', 'greedy mcts']
    for i in range(4):
        plot_path(ax, 'MCTS Method', cost, runtime,
                  network.graph.nodes, edges, network.positions)
    plt.show()


def plot_path(ax, model_name, cost, time, nodes, edges, positions):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    nx.draw(graph, positions, ax=ax, node_size=50, edge_color='0.2')
    ax.set_title('{:s} model\ncost={:.2f} time={:.4f}s'.format(model_name, cost, time))
