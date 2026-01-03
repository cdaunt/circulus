import pytest

from circulus.models import resistor, capacitor, voltage_source, inductor
from circulus.compiler import compile_netlist


def test_compile_netlist_basic(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Basic expectations
    assert isinstance(port_map, dict)
    assert "V1,p2" in port_map and "C1,p1" in port_map

    # There should be a group for each non-ground component
    group_names = set(groups.keys())
    assert {'resistor', 'capacitor', 'inductor', 'source_voltage'} <= group_names

    # Sys size should include node count + internal variables (V1 and L1 each add 1 internal)
    expected_internals = 2
    num_nets = max(port_map.values()) + 1
    assert sys_size == num_nets + expected_internals
