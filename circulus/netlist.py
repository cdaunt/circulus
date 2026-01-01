### SAX netlists will be used as much as possible in circulus;
###however, connections for node based simulators need to be handled differently.

from __future__ import annotations

import networkx as nx

from typing import Dict, Tuple
from typing import Annotated, NotRequired, TypeAlias, Union, Tuple
from typing_extensions import TypedDict

import sax
from sax.saxtypes.core import bval
from sax.saxtypes.settings import Settings
from sax.saxtypes.singlemode import InstancePort
from sax.saxtypes import Instances, Ports, Placements
import sax.saxtypes.netlist as sax_netlist
from natsort import natsorted
import matplotlib.pyplot as plt



Connections: TypeAlias = dict[InstancePort, Union[InstancePort, Tuple[InstancePort, ...]]]

CirculusNetlist = Annotated[
    TypedDict(
        "Netlist",
        {
            "instances": Instances,
            "connections": NotRequired[Connections],
            "ports": Ports,
            "placements": NotRequired[Placements],
            "settings": NotRequired[Settings],
        },
    ),
    bval(sax_netlist.val_netlist),
]
"""A complete netlist definition for an optical circuit.

Contains all information needed to define a circuit: instances,
connections, external ports, and optional placement/settings.

Attributes:
    instances: The component instances in the circuit.
    connections: Point-to-point connections between instances.
    ports: Mapping of external ports to internal instance ports.
    placements: Physical placement information for instances.
    settings: Global circuit settings.
"""

Netlist = CirculusNetlist

#Monkeypatch sax.Netlist to be CirculusNetlist so that all functions using sax.Netlist
#May need to make this explicit in the future
sax_netlist.Netlist = CirculusNetlist  # type: ignore[assignment]


def build_net_map(netlist: dict) -> Tuple[Dict[str, int], int]:
    """
    Maps every port (e.g. 'R1,p1') to a generic Node Index (integer).
    Returns:
        port_to_idx: dict mapping 'Instance,Pin' -> int index
        num_nets: Total number of unique electrical nodes (excluding Ground)
    """
    g = nx.Graph()
    
    # Add connections
    for src, targets in netlist.get("connections", {}).items():
        if isinstance(targets, str):
            targets = [targets]
        for tgt in targets:
            g.add_edge(src, tgt)
            
    # Find connected components (nets)
    components = list(nx.connected_components(g))
    components.sort(key=lambda x: natsorted(list(x))[0]) # Deterministic sort

    port_to_idx = {}
    current_idx = 1 # Start at 1, 0 is reserved for Ground

    for comp in components:
        is_ground = any('GND' in node for node in comp)
        net_id = 0 if is_ground else current_idx
        
        for node in comp:
            port_to_idx[node] = net_id
            
        if not is_ground:
            current_idx += 1
            
    return port_to_idx, current_idx


def draw_circuit_graph(netlist, show=True):
    """
    Visualizes the circuit netlist as a graph.
    - Instances are large Red nodes.
    - Ports are small SkyBlue nodes attached to instances.
    - Wires are edges between Ports, labeled with the Net Index.
    """
    
    # 1. Get Connectivity Data
    # port_map maps "Instance,Pin" -> NetIndex (int)
    port_map, num_vars = build_net_map(netlist)

    G = nx.Graph()
    
    # 2. Add Instance Nodes
    # We iterate over the 'instances' dict to find all components
    instances = netlist.get("instances", {})
    for name in instances:
        if name == 'GND': 
            # Treat GND specially if desired, or as a standard instance
            G.add_node(name, color='black', size=1500, label=name)
        else:
            G.add_node(name, color='red', size=2000, label=name)

    # 3. Add Port Nodes & Internal Connections
    # Group ports by Net ID to draw wires later
    net_groups = {} 
    
    for port_str, net_idx in port_map.items():
        if ',' not in port_str:
            continue
            
        inst_name, pin_name = port_str.split(',', 1)
        
        # Add Port Node (Small)
        # We use the full string as ID, but maybe label it just with pin name
        G.add_node(port_str, color='skyblue', size=300, label=pin_name)
        
        # Internal Edge: Connect Port to its Parent Instance
        # High weight ensures they stay close in the layout
        if inst_name in G.nodes:
            G.add_edge(inst_name, port_str, weight=5, type='internal')
        
        # Organize for external wiring
        if net_idx not in net_groups:
            net_groups[net_idx] = []
        net_groups[net_idx].append(port_str)

    # 4. Add Wire Edges (External Connections)
    edge_labels = {}
    
    for net_idx, ports in net_groups.items():
        # If a net has multiple ports, connect them.
        # We connect them sequentially to form a path (P1-P2-P3) 
        # which looks cleaner than a fully connected clique.
        if len(ports) > 1:
            for i in range(len(ports) - 1):
                u, v = ports[i], ports[i+1]
                G.add_edge(u, v, weight=1, type='external')
                edge_labels[(u, v)] = str(net_idx)

    # 5. Drawing Configuration
    fig = plt.figure(figsize=(10, 8))
    
    # Compute Layout
    # k controls the distance between nodes. 
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Separate nodes by type for styling
    instance_nodes = [n for n, d in G.nodes(data=True) if d.get('color') in ['red', 'black']]
    port_nodes = [n for n, d in G.nodes(data=True) if d.get('color') == 'skyblue']

    # Draw Instances
    nx.draw_networkx_nodes(G, pos, nodelist=instance_nodes, 
                           node_color=[G.nodes[n]['color'] for n in instance_nodes], 
                           node_size=[G.nodes[n]['size'] for n in instance_nodes])
    
    nx.draw_networkx_labels(G, pos, labels={n: n for n in instance_nodes}, 
                            font_color='white', font_weight='bold')

    # Draw Ports
    nx.draw_networkx_nodes(G, pos, nodelist=port_nodes, 
                           node_color='skyblue', node_size=300)
    
    # Draw Port Labels
    port_labels = {n: G.nodes[n]['label'] for n in port_nodes}
    nx.draw_networkx_labels(G, pos, labels=port_labels, font_size=8, font_color='black')
    
    # Draw Internal Edges (Instance -> Port) - Solid, Thicker
    internal_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'internal']
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, width=2.0, alpha=0.5)

    # Draw External Edges (Wires) - Dashed or different style
    external_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'external']
    nx.draw_networkx_edges(G, pos, edgelist=external_edges, width=1.5, style='dashed', edge_color='gray')

    # Draw Net Index Labels on Wires
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue')

    ax = plt.gca()

    ax.set_title("Circuit Connectivity Graph")
    ax.axis('off')
    fig.tight_layout()

    if show:
        plt.show()
    
    return fig

#being explicit here
Port = sax.Port
Ports = sax.Ports
Net = sax.Net
Nets = sax.Nets
Placements = sax.Placements
Instances = sax.Instances
netlist = sax.netlist