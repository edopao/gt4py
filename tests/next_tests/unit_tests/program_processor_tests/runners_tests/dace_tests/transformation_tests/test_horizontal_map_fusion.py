import pytest

dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

import copy

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

import dace

import uuid
from typing import Literal, Union, overload

import dace
from dace.sdfg import nodes as dace_nodes

from typing import Callable
import numpy as np
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation

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

    for name in ["a", "b", "c", "d", "out1", "out2", "out3", "out4"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )

    sdfg.add_scalar("tmp1", dtype=dace.float64, transient=True)
    sdfg.add_scalar("tmp2", dtype=dace.float64, transient=True)
    a, b, out1, tmp1, tmp2 = (state.add_access(name) for name in ["a", "b", "out1", "tmp1", "tmp2"])

    # tasklet1, map_entry1, map_exit1 = state.add_mapped_tasklet(
    #     name="first_computation",
    #     map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
    #     inputs={"__in0": dace.Memlet("a[__i0, __i1]")},
    #     code="__out = __in0 + 1.0",
    #     outputs={"__out": dace.Memlet("out1[__i0, __i1]")},
    #     external_edges=True,
    # )

        # First independent Tasklet.
    task1 = state.add_tasklet(
        "task1_indepenent",
        inputs={
            "__in0",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `tmp1`
        },
        code="__out0 = __in0 + 3.0",
    )

    # This is the second independent Tasklet.
    task2 = state.add_tasklet(
        "task2_indepenent",
        inputs={
            "__in0",  # <- `tmp1`
            "__in1",  # <- `b[i, j]`
        },
        outputs={
            "__out0",  # <- `tmp2`
        },
        code="__out0 = __in0 + __in1",
    )

    # This is the third Tasklet, which is dependent.
    task3 = state.add_tasklet(
        "task3_dependent",
        inputs={
            "__in0",  # <- `tmp2`
            "__in1",  # <- `a[i, j]`
        },
        outputs={
            "__out0",  # <- `out1[i, j]`.
        },
        code="__out0 = __in0 + __in1",
    )

    # Now create the map
    mentry, mexit = state.add_map(
        "map",
        ndrange={"i": f"0:{N}", "j": f"0:{N}"},
    )

    # Now assemble everything.
    state.add_edge(mentry, "OUT_b", task1, "__in0", dace.Memlet("b[i, j]"))
    state.add_edge(task1, "__out0", tmp1, None, dace.Memlet("tmp1[0]"))

    state.add_edge(tmp1, None, task2, "__in0", dace.Memlet("tmp1[0]"))
    state.add_edge(mentry, "OUT_b", task2, "__in1", dace.Memlet("b[i, j]"))
    state.add_edge(task2, "__out0", tmp2, None, dace.Memlet("tmp2[0]"))

    state.add_edge(tmp2, None, task3, "__in0", dace.Memlet("tmp2[0]"))
    state.add_edge(mentry, "OUT_a", task3, "__in1", dace.Memlet("a[i, j]"))
    state.add_edge(task3, "__out0", mexit, "IN_out1", dace.Memlet("out1[i, j]"))

    state.add_edge(a, None, mentry, "IN_a", sdfg.make_array_memlet("a"))
    state.add_edge(b, None, mentry, "IN_b", sdfg.make_array_memlet("b"))
    state.add_edge(mexit, "OUT_out1", out1, None, sdfg.make_array_memlet("out1"))
    for name in ["a", "b"]:
        mentry.add_in_connector("IN_" + name)
        mentry.add_out_connector("OUT_" + name)
    mexit.add_in_connector("IN_out1")
    mexit.add_out_connector("OUT_out1")

    tasklet2, map_entry2, map_exit2 = state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[("__i0", f"0:{N/2}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("b[__i0, __i1]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("out2[__i0, __i1]")},
        external_edges=True,
    )

    tasklet3, map_entry3, map_exit3 = state.add_mapped_tasklet(
        name="third_computation",
        map_ranges=[("__i3", f"0:{N/2}"), ("__i4", f"0:{N}")],
        inputs={"__in0": dace.Memlet("c[__i3, __i4]"), "__in1": dace.Memlet("a[__i3, __i4]")},
        code="__out = __in0 + __in1 + 1.0",
        outputs={"__out": dace.Memlet("out3[__i3, __i4]")},
        external_edges=True,
    )

    tasklet4, map_entry4, map_exit4 = state.add_mapped_tasklet(
        name="fourth_computation",
        map_ranges=[("__i3", f"{N/4}:{N}"), ("__i4", f"0:{N}")],
        inputs={"__in0": dace.Memlet("d[__i3, __i4]"), "__in1": dace.Memlet("a[__i3, __i4]")},
        code="__out = __in0 + __in1 + 2.0",
        outputs={"__out": dace.Memlet("out4[__i3, __i4]")},
        external_edges=True,
    )

    existing_access_nodes = {}

    for access_node in state.nodes():
        if isinstance(access_node, dace_nodes.AccessNode):
            if access_node.label not in existing_access_nodes:
                existing_access_nodes[access_node.label] = access_node
            else:
                print(f"Duplicate access node found: {access_node.label}")
                edges_for_removal = []
                for edge in state.in_edges(access_node):
                    print(f"Removing edge {edge} from {access_node.label}")
                    edges_for_removal.append(edge)
                    state.add_edge(edge.src, edge.src_conn, existing_access_nodes[access_node.label], edge.dst_conn, copy.deepcopy(edge.data))
                for edge in state.out_edges(access_node):
                    print(f"Removing edge {edge} from {access_node.label}")
                    edges_for_removal.append(edge)
                    state.add_edge(existing_access_nodes[access_node.label], edge.src_conn, edge.dst, edge.dst_conn, copy.deepcopy(edge.data))
                for edge in edges_for_removal:
                    state.remove_edge(edge)
                state.remove_node(access_node)

    dace_propagation.propagate_states(sdfg)
    sdfg.validate()

    return sdfg

def test_vertical_map_fusion():
    sdfg = _make_serial_sdfg_1(20)
    sdfg.view()
    import pdb; pdb.set_trace()
    _ = gtx_transformations.gt_horizontal_map_fusion(
        sdfg=sdfg,
        run_simplify=False,
        validate=False,
        validate_all=False,
    )
    sdfg.view()

if __name__ == "__main__":
    test_vertical_map_fusion()