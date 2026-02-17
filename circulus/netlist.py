"""Circulus netlists.

SAX netlists will be used as much as possible in circulus; 
however, connections for node based simulators need to be handled slightly differently.
"""


from __future__ import annotations

from typing import Annotated, NotRequired, TypeAlias, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sax
import sax.saxtypes.netlist as sax_netlist
from natsort import natsorted
from sax.saxtypes import Instances, Placements, Ports
from sax.saxtypes.core import bval
from sax.saxtypes.settings import Settings
from sax.saxtypes.singlemode import InstancePort
from typing_extensions import TypedDict

Connections: TypeAlias = dict[
    InstancePort, Union[InstancePort, tuple[InstancePort, ...]]
]

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

# Monkeypatch sax.Netlist to be CirculusNetlist so that all functions using sax.Netlist
# May need to make this explicit in the future
sax_netlist.Netlist = CirculusNetlist  # type: ignore[assignment]


def build_net_map(netlist: dict) -> tuple[dict[str, int], int]:
    """Maps every port (e.g. 'R1,p1') to a generic Node Index (integer).

    Returns:
        port_to_idx: dict mapping 'Instance,Pin' -> int index
        num_nets: Total number of unique electrical nodes (excluding Ground).

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
    components.sort(key=lambda x: natsorted(list(x))[0])  # Deterministic sort

    port_to_idx = {}
    current_idx = 1  # Start at 1, 0 is reserved for Ground

    for comp in components:
        is_ground = any("GND" in node for node in comp)
        net_id = 0 if is_ground else current_idx

        for node in comp:
            port_to_idx[node] = net_id

        if not is_ground:
            current_idx += 1

    return port_to_idx, current_idx


def draw_circuit_graph(  # noqa: C901, PLR0912, PLR0915
    netlist: dict[str, dict],
    layout_attempts: int = 10,
    *,
    show: bool = True,
) -> mpl.figure.Figure:
    """Visualize a circuit netlist as a connectivity graph.

    Nodes are split into two categories:
      - Instance nodes: large circles (red for components, black for GND)
        representing circuit components.
      - Port nodes: small skyblue circles representing the pins on each
        component. Each port is drawn close to its parent instance.

    Edges are split into two categories:
      - Internal edges: solid lines connecting each port to its parent instance.
      - External edges (wires): dashed gray lines connecting ports that share
        the same net, labelled with the net index.

    The layout is computed by running ``networkx.spring_layout`` up to
    ``layout_attempts`` times with different seeds. Each candidate layout is
    scored by counting proper edge-segment crossings, and the layout with the
    fewest crossings is used. Port nodes are warm-started near their parent
    instance to encourage tight visual clustering regardless of which seed wins.

    Args:
        netlist: Circuit description dict. Must contain an ``"instances"`` key
            mapping component names to their data. Connectivity is derived via
            ``build_net_map``, which returns a port map of the form
            ``"InstanceName,PinName" -> net_index``.
        show: If ``True``, call ``plt.show()`` before returning. Set to
            ``False`` when embedding the figure in a larger application or
            when running in a non-interactive environment.
        layout_attempts: Number of spring-layout seeds to try. The candidate
            with the fewest edge crossings is kept. Higher values improve
            crossing minimisation at the cost of extra compute time.
            Defaults to ``10``; values between ``20`` and ``30`` are
            reasonable for larger netlists.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the rendered graph.

    """
    # 1. Get Connectivity Data
    port_map, _= build_net_map(netlist)

    G = nx.Graph()

    # 2. Add Instance Nodes
    instances = netlist.get("instances", {})
    for name in instances:
        if name == "GND":
            G.add_node(name, color="black", size=1500, label=name)
        else:
            G.add_node(name, color="red", size=2000, label=name)

    # 3. Add Port Nodes & Internal Connections
    net_groups = {}

    for port_str, net_idx in port_map.items():
        if "," not in port_str:
            continue

        inst_name, pin_name = port_str.split(",", 1)

        G.add_node(
            port_str, color="skyblue", size=300, label=pin_name, parent=inst_name
        )

        if inst_name in G.nodes:
            G.add_edge(inst_name, port_str, weight=10, type="internal")

        if net_idx not in net_groups:
            net_groups[net_idx] = []
        net_groups[net_idx].append(port_str)

    # 4. Add Wire Edges (External Connections)
    edge_labels = {}

    for net_idx, ports in net_groups.items():
        if len(ports) > 1:
            for i in range(len(ports) - 1):
                u, v = ports[i], ports[i + 1]
                G.add_edge(u, v, weight=1, type="external")
                edge_labels[(u, v)] = str(net_idx)

    # 5. Build a smart initial position: instances spread out, ports near their parent
    def make_initial_pos(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        pos = {}
        instance_nodes = [
            n for n, d in G.nodes(data=True) if d.get("color") in ["red", "black"]
        ]

        # Place instances on a rough grid / random spread
        for name in instance_nodes:
            pos[name] = rng.uniform(-1, 1, size=2)

        # Place each port with a small random offset from its parent
        for n, d in G.nodes(data=True):
            if d.get("color") == "skyblue":
                parent = d.get("parent")
                if parent and parent in pos:
                    pos[n] = pos[parent] + rng.uniform(-0.1, 0.1, size=2)
                else:
                    pos[n] = rng.uniform(-1, 1, size=2)
        return pos

    # 6. Count edge crossings for a given layout (2D segments intersection test)
    def count_crossings(pos: dict[str, np.ndarray]) -> int:

        def segments_intersect(
            p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
        ) -> bool:

            def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
                return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)

            return bool(((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) 
                        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)))

        edges = list(G.edges())
        crossings = 0
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                u1, v1 = edges[i]
                u2, v2 = edges[j]
                # Skip pairs that share a node (adjacent edges always "meet", not cross)
                if u1 in (u2, v2) or v1 in (u2, v2):
                    continue
                p1, p2 = pos[u1], pos[v1]
                p3, p4 = pos[u2], pos[v2]
                if segments_intersect(p1, p2, p3, p4):
                    crossings += 1
        return crossings

    # 7. Run layout multiple times and keep the best result
    best_pos = None
    best_crossings = float("inf")

    for attempt in range(layout_attempts):
        seed = attempt  # deterministic across runs with same layout_attempts
        init_pos = make_initial_pos(seed)
        candidate_pos = nx.spring_layout(
            G,
            pos=init_pos,  # warm start near the desired structure
            fixed=None,  # let everything move, but start well
            k=0.5,
            iterations=80,  # more iterations since we have a good warm start
            weight="weight",
            seed=seed,
        )
        crossings = count_crossings(candidate_pos)
        if crossings < best_crossings:
            best_crossings = crossings
            best_pos = candidate_pos

    pos = best_pos

    # 8. Drawing Configuration
    fig = plt.figure(figsize=(10, 8))

    instance_nodes = [
        n for n, d in G.nodes(data=True) if d.get("color") in ["red", "black"]
    ]
    port_nodes = [n for n, d in G.nodes(data=True) if d.get("color") == "skyblue"]

    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=instance_nodes,
        node_color=[G.nodes[n]["color"] for n in instance_nodes],
        node_size=[G.nodes[n]["size"] for n in instance_nodes],
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={n: n for n in instance_nodes},
        font_color="white",
        font_weight="bold",
    )

    nx.draw_networkx_nodes(
        G, pos, nodelist=port_nodes, node_color="skyblue", node_size=300
    )

    port_labels = {n: G.nodes[n]["label"] for n in port_nodes}
    nx.draw_networkx_labels(G, pos, labels=port_labels, font_size=8, font_color="black")

    internal_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("type") == "internal"
    ]
    nx.draw_networkx_edges(G, pos, edgelist=internal_edges, width=2.0, alpha=0.5)

    external_edges = [
        (u, v) for u, v, d in G.edges(data=True) if d.get("type") == "external"
    ]
    nx.draw_networkx_edges(
        G, pos, edgelist=external_edges, width=1.5, style="dashed", edge_color="gray"
    )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue")

    ax = plt.gca()
    ax.set_title("Circuit Connectivity Graph")
    ax.axis("off")
    fig.tight_layout()

    if show:
        plt.show()

    return fig


# being explicit here
Port = sax.Port
Ports = sax.Ports
Net = sax.Net
Nets = sax.Nets
Placements = sax.Placements
Instances = sax.Instances
netlist = sax.netlist
