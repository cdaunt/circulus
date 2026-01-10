import sys
from pathlib import Path
import pytest
import jax

# Ensure project root is on sys.path so tests can import the local package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared fixtures for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_lrc_netlist():
    """Returns (net_dict, models_map) for a small LRC example."""
    from circulus.components import Resistor, Capacitor, Inductor, VoltageSource

    models_map = {
        'resistor': Resistor,
        'capacitor': Capacitor,
        'inductor': Inductor,
        'source_voltage': VoltageSource,
        'ground': lambda: 0
    }

    net_dict = {
        "instances": {
            "GND": {"component":"ground"},
            "V1": {"component":"source_voltage", "settings":{"V": 5.0, "delay":0.2e-8}},
            "R1": {"component":"resistor", "settings":{"R": 10.0}},
            "C1": {"component":"capacitor", "settings":{"C": 1e-11}},
            "L1": {"component":"inductor", "settings":{"L": 5e-9}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "C1,p2"),
            "V1,p2": "R1,p1",
            "R1,p2": "L1,p1",
            "L1,p2": "C1,p1",
        },
    }

    return net_dict, models_map

@pytest.fixture
def simple_optical_netlist():
    from circulus.components import Resistor
    from circulus.photonic_components import OpticalWaveguide, OpticalSourcePulse

    models_map = {'waveguide': OpticalWaveguide, 'source': OpticalSourcePulse, 'resistor': Resistor, 'ground': lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {"component": "source", "settings": {"power": 1.0 + 0j, "delay": 0.1e-9}},
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0}},
            "R1": {"component": "resistor", "settings": {"R": 1.0}} # circulus.components.Resistor defaults to 1k, we set 1.0
        },
        "connections": {"GND,p1": ("I1,p2", "R1,p2"), "I1,p1": "WG1,p1", "WG1,p2": "R1,p1"}
    }

    return net_dict, models_map
