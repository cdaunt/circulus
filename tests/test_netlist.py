import pytest
from matplotlib.figure import Figure

from circulus.netlist import draw_circuit_graph


def test_draw_circuit_graph_returns_figure(simple_lrc_netlist):
    net_dict, _ = simple_lrc_netlist
    fig = draw_circuit_graph(net_dict, show=False)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1
