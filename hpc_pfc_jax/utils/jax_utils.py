"""JAX utility functions for handling pytrees with non-array leaves."""

import equinox as eqx
import equinox.internal as eqxi
import jax


def filter_scan(f, init, xs=None, length=None, reverse=False, unroll=1, _split_transpose=False):
    """Wrapper on top of jax.lax.scan which allows pytrees with non-array leaves.

    NOTE: When calling jax.lax.scan directly, any non-ndarray array leaves that can be
    converted to traced arrays will get converted. With filter_scan, they will just be filtered
    out.

    Example:
        >>> @jdc.pytree_dataclass
        >>> class Carry:
        >>>     step: int
        >>>
        >>> def _update_step(state, unused):
        >>>     step = state.step + 1
        >>>     y = None
        >>>     return jdc.replace(state, step=step), y
        >>>
        >>> init = Carry(step=0)
        >>> carry, _ = filter_scan(_update_step, init=init, xs=None, length=100)
        >>> # 0, since we remove the static output from the carry before passing to next step
        >>> assert carry.step == 0
    """
    init, static = eqx.partition(init, eqx.is_array)

    def scan_closure(carry, x):
        carry = eqx.combine(carry, static)
        carry, y = f(carry, x)
        carry, _ = eqx.partition(carry, eqx.is_array)
        return carry, y

    carry, ys = jax.lax.scan(
        scan_closure, init=init, xs=xs, length=length, reverse=reverse, unroll=unroll, _split_transpose=_split_transpose
    )
    return eqx.combine(carry, static), ys


def filter_cond(pred, true_fun, false_fun, *operands):
    """Wrapper on top of jax.lax.cond which allows pytrees with non-array leaves.

    Note: See filter_scan for understanding differences between
    these filter APIs and the underlying jax.lax behaviour.
    """
    dynamic, static = eqx.partition(operands, eqx.is_array)

    def _true_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = true_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    def _false_fun(_dynamic):
        _operands = eqx.combine(_dynamic, static)
        _out = false_fun(*_operands)
        _dynamic_out, _static_out = eqx.partition(_out, eqx.is_array)
        return _dynamic_out, eqxi.Static(_static_out)

    dynamic_out, static_out = jax.lax.cond(pred, _true_fun, _false_fun, dynamic)
    return eqx.combine(dynamic_out, static_out.value)
