import sax
import jax.numpy as jnp
from sax import netlist
from sax.circuit import (
    _create_dag,
    _find_leaves,
    _find_root,
    _flat_circuit,
    _validate_models,
    draw_dag,
)
import matplotlib.pyplot as plt
from sax.netlist import _get_connectivity_netlist, _get_connectivity_graph
from circulus.models import resistor, capacitor, voltage_source, inductor
import networkx as nx
from typing import Dict
from sax import Model, scoo, SCoo
from sax.netlist import Component

def analyze_instances_circulus(
    instances: Dict[str, Component],
    models: Dict[str, Model],
) -> Dict[str, SCoo]:
    instances, instances_old = {}, instances
    for k, v in instances_old.items():
        if not isinstance(v, Component):
            v = Component(**v)
        instances[k] = v
    model_names = set()
    for i in instances.values():
        model_names.add(i.component)
    dummy_models = {k: scoo(models[k]()) for k in model_names}
    dummy_instances = {}
    for k, i in instances.items():
        dummy_instances[k] = dummy_models[i.component]
    return dummy_instances

if __name__ == "__main__":

    net_dict = {
    "instances": {
        "GND": {"component":"ground"},
        "V1": {"component":"source_voltage"},
        "R1": {"component":"resistor"},
        "C1": {"component":"capacitor"},
        "L1": {"component":"inductor"}
    },
    "connections": {
        "GND,p1": "V1,p1",
        "R1,p1": "V1,p2",
        "R1,p2": "C1,p1",
        "C1,p2": "L1,p1",
    },
    "ports": {
        "out": "L1,p2",
    },
    "models":{'resistor': resistor,
              'capacitor': capacitor,
              'inductor': inductor,
              'source_voltage': voltage_source,
                'ground': sax.passthru
              }
    }
    

    net = netlist(net_dict)
    dag = _get_connectivity_graph(net_dict)
    #draw_dag(dag)
    #nx.draw(dag, with_labels=True)
    #plt.show()

    analyzed_instances = analyze_instances_circulus(net_dict['instances'], net_dict['models'])
    #analyzed_circuit = sax.backends.analyze_circuit_c

