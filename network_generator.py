import numpy as np
import networkx as nx
from collections import defaultdict

class NetworkGenerator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_scale_free_with_communities(self, num_nodes: int, m: int, num_communities: int, p_inter_community: float, agent_ideologies: np.ndarray) -> nx.Graph:
        """
        Generates a scale-free network with community structure.
        This approach combines Barabasi-Albert for scale-free properties within communities
        and then connects communities with a given probability.

        num_nodes: Total number of nodes in the network.
        m: Number of edges to attach from a new node to existing nodes in Barabasi-Albert model.
        num_communities: Desired number of communities.
        p_inter_community: Probability of forming an edge between nodes in different communities.
        agent_ideologies: Array of agent ideologies (or any attribute for homophily) for rewiring.
        """
        print(f"Generating scale-free network with {num_communities} communities...")
        graph = nx.Graph()
        nodes_per_community = num_nodes // num_communities
        community_nodes = defaultdict(list)

        # 1. Generate scale-free subgraphs for each community
        for i in range(num_communities):
            start_node = i * nodes_per_community
            end_node = (i + 1) * nodes_per_community if i < num_communities - 1 else num_nodes
            current_community_nodes = list(range(start_node, end_node))
            community_nodes[i] = current_community_nodes

            if len(current_community_nodes) > m:
                subgraph = nx.barabasi_albert_graph(len(current_community_nodes), m, seed=self.rng.integers(0, 100000))
                # Relabel nodes to fit into the global node numbering
                mapping = {old_node: new_node for old_node, new_node in zip(range(len(current_community_nodes)), current_community_nodes)}
                subgraph = nx.relabel_nodes(subgraph, mapping)
                graph.add_edges_from(subgraph.edges())
            else:
                # Handle small communities, e.g., just add nodes without edges or with minimal edges
                graph.add_nodes_from(current_community_nodes)
                if len(current_community_nodes) > 1:
                    # Add a few random edges if community is too small for BA
                    for _ in range(len(current_community_nodes) // 2):
                        u, v = self.rng.choice(current_community_nodes, 2, replace=False)
                        graph.add_edge(u, v)

        # 2. Add inter-community edges with probability p_inter_community
        print("Adding inter-community edges...")
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if not graph.has_edge(i, j):
                    # Check if they are in different communities
                    comm_i = i // nodes_per_community
                    comm_j = j // nodes_per_community
                    if comm_i != comm_j:
                        if self.rng.random() < p_inter_community:
                            graph.add_edge(i, j)

        # 3. Apply homophily rewiring (as previously discussed, but now on a structured network)
        print("Applying homophily rewiring...")
        # This is a simplified example. A more robust implementation would iterate through edges
        # and rewire based on agent attributes (e.g., ideology).
        edges_to_rewire = []
        for u, v in list(graph.edges()): # Iterate over a copy as edges might be removed
            # Check if agents are dissimilar based on ideology and if they are within the same community
            # or if they are inter-community edges that should be rewired
            if abs(agent_ideologies[u] - agent_ideologies[v]) > 0.5: # Example threshold for dissimilarity
                if self.rng.random() < 0.1: # rewire_prob for homophily
                    edges_to_rewire.append((u, v))

        for u, v in edges_to_rewire:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
                # Find a new node w that is similar to u but not connected to u
                potential_new_neighbors = [node_id for node_id in graph.nodes() if 
                                           abs(agent_ideologies[u] - agent_ideologies[node_id]) < 0.2 and # Example similarity threshold
                                           not graph.has_edge(u, node_id) and 
                                           u != node_id]
                if potential_new_neighbors:
                    w = self.rng.choice(potential_new_neighbors)
                    graph.add_edge(u, w)

        print(f"Final network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
        return graph

    def generate_multiplex_network(self, num_nodes: int, layer_configs: list[dict], agent_attributes: dict) -> dict:
        """
        Generates a multiplex network with different layers (e.g., social media platforms).
        Each layer can have its own structure (e.g., one scale-free, one random).

        num_nodes: Total number of nodes.
        layer_configs: List of dictionaries, each specifying a layer:
            - 'name': Name of the layer (e.g., 'Facebook', 'Twitter')
            - 'type': 'scale_free' or 'random'
            - 'params': Dictionary of parameters for the specific network type (e.g., {'m': 3} for BA, {'p': 0.1} for Erdos-Renyi)
        agent_attributes: Dictionary of agent attributes (e.g., {'ideology': np.array([...])}) for homophily.
        """
        print("Generating multiplex network...")
        multiplex_graph = {}
        for config in layer_configs:
            layer_name = config["name"]
            layer_type = config["type"]
            layer_params = config["params"]
            print(f"  - Creating layer: {layer_name} ({layer_type})")

            if layer_type == "scale_free":
                m = layer_params.get("m", 3)
                graph = nx.barabasi_albert_graph(num_nodes, m, seed=self.rng.integers(0, 100000))
            elif layer_type == "random":
                p = layer_params.get("p", 0.01)
                graph = nx.erdos_renyi_graph(num_nodes, p, seed=self.rng.integers(0, 100000))
            else:
                raise ValueError(f"Unsupported network type: {layer_type}")

            # Apply homophily if ideology is available and configured
            if "ideology" in agent_attributes and layer_params.get("apply_homophily", False):
                print(f"    - Applying homophily to layer {layer_name}...")
                rewire_prob = layer_params.get("rewire_prob", 0.1)
                homophily_strength = layer_params.get("homophily_strength", 0.5)
                edges_to_rewire = []
                for u, v in list(graph.edges()):
                    if self.rng.random() < rewire_prob:
                        if abs(agent_attributes["ideology"][u] - agent_attributes["ideology"][v]) > homophily_strength:
                            edges_to_rewire.append((u, v))

                for u, v in edges_to_rewire:
                    if graph.has_edge(u, v):
                        graph.remove_edge(u, v)
                        potential_new_neighbors = [node_id for node_id in graph.nodes() if 
                                                   abs(agent_attributes["ideology"][u] - agent_attributes["ideology"][node_id]) < homophily_strength and 
                                                   not graph.has_edge(u, node_id) and 
                                                   u != node_id]
                        if potential_new_neighbors:
                            w = self.rng.choice(potential_new_neighbors)
                            graph.add_edge(u, w)

            multiplex_graph[layer_name] = graph
            print(f"    - Layer {layer_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")

        return multiplex_graph


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    net_gen = NetworkGenerator(rng)

    num_agents = 1000
    agent_ideologies = rng.normal(0, 1, num_agents) # Dummy ideologies

    # Example 1: Scale-free network with communities
    print("\n--- Testing Scale-Free Network with Communities ---")
    community_graph = net_gen.generate_scale_free_with_communities(
        num_nodes=num_agents, m=3, num_communities=5, p_inter_community=0.01,
        agent_ideologies=agent_ideologies
    )
    print(f"Generated Community Graph: {community_graph.number_of_nodes()} nodes, {community_graph.number_of_edges()} edges.")

    # Example 2: Multiplex network
    print("\n--- Testing Multiplex Network ---")
    layer_configs = [
        {"name": "Facebook", "type": "scale_free", "params": {"m": 5, "apply_homophily": True, "rewire_prob": 0.1, "homophily_strength": 0.5}},
        {"name": "Twitter", "type": "random", "params": {"p": 0.005, "apply_homophily": True, "rewire_prob": 0.2, "homophily_strength": 0.3}}
    ]
    agent_attrs = {"ideology": agent_ideologies}
    multiplex_net = net_gen.generate_multiplex_network(
        num_nodes=num_agents, layer_configs=layer_configs, agent_attributes=agent_attrs
    )
    for layer_name, graph in multiplex_net.items():
        print(f"Layer \'{layer_name}\': {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")


