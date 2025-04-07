import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

import dace

import uuid
from typing import Literal, Union, overload

import dace
from dace.sdfg import nodes as dace_nodes


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[False],
) -> int: ...


@overload
def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: Literal[True],
) -> list[dace_nodes.Node]: ...


def count_nodes(
    graph: Union[dace.SDFG, dace.SDFGState],
    node_type: tuple[type, ...] | type,
    return_nodes: bool = False,
) -> Union[int, list[dace_nodes.Node]]:
    """Counts the number of nodes in of a particular type in `graph`.

    If `graph` is an SDFGState then only count the nodes inside this state,
    but if `graph` is an SDFG count in all states.

    Args:
        graph: The graph to scan.
        node_type: The type or sequence of types of nodes to look for.
    """

    states = graph.states() if isinstance(graph, dace.SDFG) else [graph]
    found_nodes: list[dace_nodes.Node] = []
    for state_nodes in states:
        for node in state_nodes.nodes():
            if isinstance(node, node_type):
                found_nodes.append(node)
    if return_nodes:
        return found_nodes
    return len(found_nodes)


def unique_name(name: str) -> str:
    """Adds a unique string to `name`."""
    return f"{name}_{str(uuid.uuid1()).replace('-', '_')}"


def _make_serial_sdfg_1(
    N: str | int,
) -> dace.SDFG:
    """Create the "serial_1_sdfg".

    This is an SDFG with a single state containing two maps. It has the input
    `a` and `b`, each two dimensional arrays, with size `0:N`.
    The two maps add 1 to the input and write them into `out1` and `out2` corespondingly.
    The second map only adds 1 to half of the input `b` and writes it into `out2`.

    Args:
        N: The size of the arrays.
    """
    shape = (N, N)
    sdfg = dace.SDFG(unique_name("serial_sdfg1"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "out1", "out2"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("out1[__i0, __i1]")},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[("__i0", f"0:{N/2}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("b[__i0, __i1]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("out2[__i0, __i1]")},
        external_edges=True,
    )

    return sdfg

def test_vertical_map_fusion():
    sdfg = _make_serial_sdfg_1(20)
    _ = gtx_transformations.gt_horizontal_map_fusion(
        sdfg=sdfg,
        run_simplify=False,
        validate=False,
        validate_all=False,
    )
    sdfg.view()

if __name__ == "__main__":
    test_vertical_map_fusion()