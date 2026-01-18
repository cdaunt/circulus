import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple, ClassVar, Any, Union, Dict
from collections import namedtuple

class CircuitComponent(eqx.Module):
    # Standard Metadata
    ports: ClassVar[Tuple[str, ...]] = ()
    states: ClassVar[Tuple[str, ...]] = ()
    
    # Internal Caches
    _VarsType_P: ClassVar[Any] = None
    _VarsType_S: ClassVar[Any] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Create lightweight named tuples for dot-access (v.p1)
        if cls.ports:
            cls._VarsType_P = namedtuple("Ports", cls.ports)
        if cls.states:
            cls._VarsType_S = namedtuple("States", cls.states)

    def physics(self, v, s, t) -> Tuple[Dict, Dict]:
        """
        User implements this.
        Returns:
            (dict, dict): (Resistive f, Reactive q).
        """
        raise NotImplementedError

    @classmethod
    def solver_bridge(cls, vars_vec: jax.Array, instance: "CircuitComponent", t: float):
        """
        Fixed Signature: (Vector, ComponentInstance, Time)
        Matches your call: physics_func(v, p, t=t0)
        """
        # 1. Unpack Vector into Dot-Accessible Objects
        n_p = len(cls.ports)
        
        # Now 'vars_vec' is correctly the JAX Array
        v_raw, s_raw = vars_vec[:n_p], vars_vec[n_p:]
        
        # Handle empty ports/states safely
        v = cls._VarsType_P(*v_raw) if cls.ports else ()
        s = cls._VarsType_S(*s_raw) if cls.states else ()

        # 2. Call the Instance Method
        # We use 'instance' (the VoltageSource object) as 'self'
        f_dict, q_dict = instance.physics(v, s, t)

        # 3. Flatten Dictionaries to Arrays
        full_keys = cls.ports + cls.states
        
        if not full_keys:
            return jnp.zeros(0), jnp.zeros(0)

        f_vec = jnp.stack([f_dict.get(k, 0.0) for k in full_keys])
        q_vec = jnp.stack([q_dict.get(k, 0.0) for k in full_keys])

        return f_vec, q_vec