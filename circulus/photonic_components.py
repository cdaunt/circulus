import jax
import jax.numpy as jnp
import jax.nn as jnn
from circulus.base_component import component, source, Signals, States
from circulus.s_transforms import s_to_y

# ===========================================================================
# Passive Optical Components (S-Matrix based)
# ===========================================================================

@component(ports=("p1", "p2"))
def OpticalWaveguide(
    signals: Signals, 
    s: States, 
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    center_wavelength_nm: float = 1310.0,
    wavelength_nm: float = 1310.0
):
    d_lam = wavelength_nm - center_wavelength_nm
    slope = (neff - n_group) / center_wavelength_nm
    n_eff_disp = neff + slope * d_lam
    
    # Phase calculation
    phi = 2.0 * jnp.pi * n_eff_disp * (length_um / wavelength_nm) * 1000.0
    
    # Loss calculation
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    
    # S-Matrix construction
    T = T_mag * jnp.exp(-1j * phi)
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    
    # Convert to Admittance (Y)
    Y = s_to_y(S)
    
    # Calculate Currents (I = Y * V)
    # Note: Explicit complex cast ensures JAX treats the interaction as complex
    v_vec = jnp.array([signals.p1, signals.p2], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    
    return {"p1": i_vec[0], "p2": i_vec[1]}, {}


@component(ports=("grating", "waveguide"))
def Grating(
    signals: Signals, 
    s: States, 
    center_wavelength_nm: float = 1310.0,
    peak_loss_dB: float = 0.0,
    bandwidth_1dB: float = 20.0,
    wavelength_nm: float = 1310.0
):
    delta = wavelength_nm - center_wavelength_nm
    excess_loss = (delta / (0.5 * bandwidth_1dB))**2
    loss_dB = peak_loss_dB + excess_loss
    
    T = 10.0 ** (-loss_dB / 20.0)
    # Numerical stability clip
    T = jnp.minimum(T, 0.9999)
    
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    Y = s_to_y(S)
    
    v_vec = jnp.array([signals.grating, signals.waveguide], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    
    return {"grating": i_vec[0], "waveguide": i_vec[1]}, {}


@component(ports=("p1", "p2", "p3"))
def Splitter(
    signals: Signals, 
    s: States, 
    split_ratio: float = 0.5
):
    r = jnp.sqrt(split_ratio)
    tc = jnp.sqrt(1.0 - split_ratio)
    
    S = jnp.array([
        [0.0, r, 1j*tc],
        [r, 0.0, 0.0],
        [1j*tc, 0.0, 0.0]
    ], dtype=jnp.complex128)
    
    Y = s_to_y(S)
    
    v_vec = jnp.array([signals.p1, signals.p2, signals.p3], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    
    return {"p1": i_vec[0], "p2": i_vec[1], "p3": i_vec[2]}, {}


# ===========================================================================
# Optical Sources
# ===========================================================================

@component(ports=("p1", "p2"), states=("i_src",))
def OpticalSource(
    signals: Signals, 
    s: States, 
    power: float = 1.0,
    phase: float = 0.0
):
    """Phasor Source for DC/AC analysis (Time-Invariant)."""
    v_val = jnp.sqrt(power) * jnp.exp(1j * phase)
    constraint = (signals.p1 - signals.p2) - v_val
    
    return {
        "p1": s.i_src, 
        "p2": -s.i_src,
        "i_src": constraint
    }, {}


@source(ports=("p1", "p2"), states=("i_src",))
def OpticalSourcePulse(
    signals: Signals, 
    s: States, 
    t: float,
    power: float = 1.0,
    phase: float = 0.0,
    delay: float = 0.2e-9,
    rise: float = 0.05e-9
):
    """Transient Source with sigmoid rise (Time-Dependent)."""
    val = jnp.sqrt(power) * jnn.sigmoid((t - delay) / rise)
    v_val = val * jnp.exp(1j * phase)
    
    constraint = (signals.p1 - signals.p2) - v_val
    
    return {
        "p1": s.i_src, 
        "p2": -s.i_src, 
        "i_src": constraint
    }, {}