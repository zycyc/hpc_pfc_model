"""Shape annotation and transform utilities."""

import ast
import keyword
import operator as op
from ast import NodeVisitor
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, ParamSpec, TypeVar

import jax.numpy as jnp

_keywords = set(keyword.kwlist)

_axis_type = int | str | Callable
_shape_type = tuple[_axis_type, ...]
_shapes_type = tuple[_shape_type]
_inputs_type = ParamSpec("_inputs_type")
_outputs_type = TypeVar("_outputs_type", bound=jnp.ndarray | Sequence[jnp.ndarray])

_operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.floordiv,
}


class _EvalVisitor(NodeVisitor):
    def __init__(self, **kwargs):
        self._namespace = kwargs

    def visit_Name(self, node):
        return self._namespace[node.id]

    def visit_Constant(self, node: ast.Constant):
        return node.value

    def visit_NameConstant(self, node: ast.NameConstant):
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp):
        raise NotImplementedError("UnaryOps not supported")

    def visit_BinOp(self, node: ast.BinOp):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        operator_type = type(node.op)
        if operator_type not in _operators:
            raise ValueError(f"Unsupported binary operator: {operator_type}")
        return _operators[operator_type](lhs, rhs)

    def generic_visit(self, node):
        raise ValueError("malformed node or string: " + repr(node))


def _parse_annotation(annotation: str) -> tuple[_shapes_type, _shapes_type]:
    annotation = annotation.strip()
    try:
        input_shapes, output_shapes = annotation.split("->")
    except ValueError as err:
        raise ValueError("Invalid transformation annotation: must be '{{input_shapes}} -> {{output_shapes}}'") from err

    input_shapes = input_shapes.strip()
    output_shapes = output_shapes.strip()
    if not len(input_shapes) or not len(output_shapes):
        raise ValueError("Invalid transformation annotation: must have input and output shapes")

    for char in input_shapes + output_shapes:
        if not char.isalnum() and char not in (",", "(", ")", " ", "*", "-", "/", "+", "_"):
            raise ValueError(f"Invalid transformation annotation: invalid character: {char}")

    def _validate_and_standardize_shapes(node: ast.Tuple) -> ast.Tuple:
        at_least_one_nested_tuple = False
        at_least_one_non_tuple = False
        for elt in node.elts:
            if isinstance(elt, ast.Tuple):
                at_least_one_nested_tuple = True
                for subelt in elt.elts:
                    if isinstance(subelt, ast.Tuple):
                        raise ValueError("Invalid transformation annotation. Nested tuples are not allowed.")
            else:
                at_least_one_non_tuple = True

        if at_least_one_nested_tuple and at_least_one_non_tuple:
            raise ValueError("Invalid transformation annotation. Cannot mix tuples and non-tuples in shape annotation.")

        if at_least_one_non_tuple:
            node = ast.Tuple([node])

        return node

    def _parse_shape_string(shape: str) -> tuple[Any, ...]:
        node = ast.parse(shape, mode="eval").body
        if not isinstance(node, ast.Tuple):
            raise ValueError("Invalid transformation annotation. Node must be a tuple.")

        if not node.elts:
            return ((),)

        node = _validate_and_standardize_shapes(node)

        processed = []

        def _construct_lambda(variables: list[str], bin_op: ast.BinOp, values: list[int]):
            kv = {k: v for k, v in zip(variables, values, strict=True)}
            return _EvalVisitor(**kv).visit(bin_op)

        def _process_bin_op(bin_op: ast.BinOp):
            variables = []
            for node in ast.walk(bin_op):
                if isinstance(node, ast.Name):
                    variables.append(node.id)

            return partial(_construct_lambda, variables, bin_op)

        for elt in node.elts:
            assert isinstance(elt, ast.Tuple | ast.List)
            tuple_content = []
            for subelt in elt.elts:
                if isinstance(subelt, ast.Constant):
                    ret = subelt.value
                elif isinstance(subelt, ast.Name):
                    ret = subelt.id
                elif isinstance(subelt, ast.BinOp):
                    ret = _process_bin_op(subelt)
                else:
                    raise ValueError(f"Invalid shape annotation. Illegal operation: {subelt}")
                tuple_content.append(ret)
            processed.append(tuple(tuple_content))

        return tuple(processed)

    try:
        ins = _parse_shape_string(input_shapes)
        outs = _parse_shape_string(output_shapes)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError, AssertionError) as err:
        raise ValueError(
            f"Invalid transformation annotation due to malformed annotation: {type(err).__name__}({err.args[0]})"
        ) from err

    return ins, outs


def _replace_keywords(annotation: str) -> tuple[str, dict[str, str]]:
    import re

    replaced_keywords = {}
    for kw in _keywords:
        pattern = rf"\b{kw}\b(?![a-zA-Z0-9])"
        if re.search(pattern, annotation):
            new_kw = f"_{kw}"
            annotation = re.sub(pattern, new_kw, annotation)
            replaced_keywords[new_kw] = kw

    return annotation, replaced_keywords


def _transform_and_check(
    transform: Callable[_inputs_type, _outputs_type],
    annotation: str,
    *args: _inputs_type.args,
    **kwargs: _inputs_type.kwargs,
) -> _outputs_type:
    annotation, replaced_keywords = _replace_keywords(annotation)
    expected_in_shapes, expected_out_shapes = _parse_annotation(annotation)

    arrays = []
    for arg in args:
        if isinstance(arg, jnp.ndarray):
            arrays.append(arg)

    for kwarg in kwargs.values():
        if isinstance(kwarg, jnp.ndarray):
            arrays.append(kwarg)

    trans_out = transform(*args, **kwargs)
    trans_out_container = trans_out
    if isinstance(trans_out, jnp.ndarray):
        trans_out_container = (trans_out,)
    trans_out_shapes = tuple(t.shape for t in trans_out_container)

    trans_in_shapes = tuple(t.shape for t in arrays)
    for recv_shape, exp_shape in zip(trans_in_shapes, expected_in_shapes, strict=True):
        if len(recv_shape) != len(exp_shape):
            raise ValueError(f"Rank of input should be {len(exp_shape)}, got {len(recv_shape)}")

    for recv_shape, exp_shape in zip(trans_out_shapes, expected_out_shapes, strict=True):
        if len(recv_shape) != len(exp_shape):
            raise ValueError(f"Rank of output should be {len(exp_shape)}, got {len(recv_shape)}")

    bound_dims = dict()
    expression_and_return_value = dict()
    for recv_shape, exp_shape in zip(trans_in_shapes, expected_in_shapes, strict=True):
        for recv_dim, exp_dim in zip(recv_shape, exp_shape, strict=True):
            if isinstance(exp_dim, int):
                if recv_dim != exp_dim:
                    raise ValueError(
                        "Mismatch between actual input shape and annotated "
                        f"input shape. Actual: {recv_shape}, annotated {exp_shape}"
                    )
            elif isinstance(exp_dim, str):
                if exp_dim not in bound_dims:
                    bound_dims[exp_dim] = recv_dim
                elif bound_dims[exp_dim] != recv_dim:
                    _exp_dim = replaced_keywords.get(exp_dim, exp_dim)
                    raise ValueError(
                        f"{_exp_dim} was already bound to {bound_dims[exp_dim]}, trying to bind to {recv_dim}"
                    )
            elif isinstance(exp_dim, Callable):
                expression_and_return_value[exp_dim] = recv_dim

    for recv_shape, exp_shape in zip(trans_out_shapes, expected_out_shapes, strict=True):
        for recv_dim, exp_dim in zip(recv_shape, exp_shape, strict=True):
            if isinstance(exp_dim, int):
                if recv_dim != exp_dim:
                    raise ValueError(
                        "Mismatch between actual output shape and annotated "
                        f"output shape. Actual: {recv_shape}, annotated {exp_shape}"
                    )
            elif isinstance(exp_dim, str):
                if exp_dim not in bound_dims:
                    bound_dims[exp_dim] = recv_dim
                elif bound_dims[exp_dim] != recv_dim:
                    _exp_dim = replaced_keywords.get(exp_dim, exp_dim)
                    raise ValueError(
                        f"{_exp_dim} was already bound to {bound_dims[exp_dim]}, trying to bind to {recv_dim}"
                    )
            elif isinstance(exp_dim, Callable):
                expression_and_return_value[exp_dim] = recv_dim

    for expression, return_value in expression_and_return_value.items():
        dims = expression.args[0]
        for dim in dims:
            if dim not in bound_dims:
                raise ValueError(f"Could not evaluate {expression} because {dim} was not bound")
        expected_return_value = expression([bound_dims[dim] for dim in dims])
        if return_value != expected_return_value:
            raise ValueError(f"Could not evaluate {expression} to {expected_return_value}")

    return trans_out


def annotate_transform(
    transform: Callable[_inputs_type, _outputs_type], annotation: str
) -> Callable[_inputs_type, _outputs_type]:
    """Annotates and checks transformations to jnp.ndarrays when the returned function is invoked.

    If annotation does not match the actual transform, raises ValueError.

    Example:
        >>> in_shape = (5, 3, 24, 24)
        >>> a = jnp.ones(in_shape)
        >>> b = annotate_transform(jnp.sum, "(b, c, h, w) -> (b, h, w)")(a, axis=1)
        >>> c = annotate_transform(jnp.sum, "(5, 3, 24, 24) -> (b, h, 24)")(a, axis=1)
        >>> assert (b == c).all()

    Note: This is expensive to do at runtime, so if using this function, make sure to jit the caller function.
    """
    return partial(_transform_and_check, transform, annotation)
