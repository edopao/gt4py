# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import copy

import numpy as np
import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes
from dace.transformation import dataflow as dace_dataflow

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

import dace


from . import util


def _make_serial_sdfg_1(
    N: str | int,
) -> dace.SDFG:
    """Create the "serial_1_sdfg".

    This is an SDFG with a single state containing two maps. It has the input
    `a` and the output `b`, each two dimensional arrays, with size `0:N`.
    The first map adds 1 to the input and writes it into `tmp`. The second map
    adds another 3 to `tmp` and writes it back inside `b`.

    Args:
        N: The size of the arrays.
    """
    shape = (N, N)
    sdfg = dace.SDFG(util.unique_name("serial_sdfg1"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "tmp"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp"].transient = True
    tmp = state.add_access("tmp")

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in0 + 1.0",
        outputs={"__out": dace.Memlet("tmp[__i0, __i1]")},
        output_nodes={tmp},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[("__i4", f"0:{N}"), ("__i6", f"0:{N}")],
        input_nodes={tmp},
        inputs={"__in0": dace.Memlet("tmp[__i4, __i6]")},
        code="__out = __in0 + 3.0",
        outputs={"__out": dace.Memlet("b[__i4, __i6]")},
        external_edges=True,
    )

    return sdfg


def _make_serial_sdfg_2(
    N: str | int,
) -> dace.SDFG:
    """Create the "serial_2_sdfg".

    The generated SDFG uses `a` and input and has two outputs `b := a + 4` and
    `c := a - 4`. There is a top map with a single Single Tasklet, that has
    two outputs, the first one computes `a + 1` and stores that in `tmp_1`.
    The second output computes `a - 1` and stores it `tmp_2`.
    Below the top map are two (parallel) map, one compute `b := tmp_1 + 3`, while
    the other compute `c := tmp_2 - 3`. This means that there are two map fusions.
    The main important thing is that, the second map fusion will involve a pure
    fusion (because the processing order is indeterministic, one does not know
    which one in advance).

    Args:
        N: The size of the arrays.
    """
    shape = (N, N)
    sdfg = dace.SDFG(util.unique_name("serial_sdfg2"))
    state = sdfg.add_state(is_start_block=True)

    for name in ["a", "b", "c", "tmp_1", "tmp_2"]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp_1"].transient = True
    sdfg.arrays["tmp_2"].transient = True
    tmp_1 = state.add_access("tmp_1")
    tmp_2 = state.add_access("tmp_2")

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("a[__i0, __i1]")},
        code="__out0 = __in0 + 1.0\n__out1 = __in0 - 1.0",
        outputs={
            "__out0": dace.Memlet("tmp_1[__i0, __i1]"),
            "__out1": dace.Memlet("tmp_2[__i0, __i1]"),
        },
        output_nodes={tmp_1, tmp_2},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        input_nodes={tmp_1},
        inputs={"__in0": dace.Memlet("tmp_1[__i0, __i1]")},
        code="__out = __in0 + 3.0",
        outputs={"__out": dace.Memlet("b[__i0, __i1]")},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        name="second_computation",
        map_ranges=[("__i3", f"0:{N}"), ("__i6", f"0:{N}")],
        input_nodes={tmp_2},
        inputs={"__in0": dace.Memlet("tmp_2[__i3, __i6]")},
        code="__out = __in0 - 3.0",
        outputs={"__out": dace.Memlet("c[__i3, __i6]")},
        external_edges=True,
    )

    return sdfg


def _make_serial_sdfg_3(
    N_input: str | int,
    N_output: str | int,
) -> dace.SDFG:
    """Creates a serial SDFG that has an indirect access Tasklet in the second map.

    The SDFG has three inputs `a`, `b` and `idx`. The first two are 1 dimensional
    arrays, and the second is am array containing integers.
    The top map computes `a + b` and stores that in `tmp`.
    The second map then uses the elements of `idx` to make indirect accesses into
    `tmp`, which are stored inside `c`.

    Args:
        N_input: The length of `a` and `b`.
        N_output: The length of `c` and `idx`.
    """
    input_shape = (N_input,)
    output_shape = (N_output,)

    sdfg = dace.SDFG(util.unique_name("serial_sdfg3"))
    state = sdfg.add_state(is_start_block=True)

    for name, shape in [
        ("a", input_shape),
        ("b", input_shape),
        ("c", output_shape),
        ("idx", output_shape),
        ("tmp", input_shape),
    ]:
        sdfg.add_array(
            name=name,
            shape=shape,
            dtype=dace.int32 if name == "idx" else dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp"].transient = True
    tmp = state.add_access("tmp")

    state.add_mapped_tasklet(
        name="first_computation",
        map_ranges=[("__i0", f"0:{N_input}")],
        inputs={
            "__in0": dace.Memlet("a[__i0]"),
            "__in1": dace.Memlet("b[__i0]"),
        },
        code="__out = __in0 + __in1",
        outputs={"__out": dace.Memlet("tmp[__i0]")},
        output_nodes={tmp},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        name="indirect_access",
        map_ranges=[("__i1", f"0:{N_output}")],
        input_nodes={tmp},
        inputs={
            "__index": dace.Memlet("idx[__i1]"),
            "__array": dace.Memlet.simple("tmp", subset_str=f"0:{N_input}", num_accesses=1),
        },
        code="__out = __array[__index]",
        outputs={"__out": dace.Memlet("c[__i1]")},
        external_edges=True,
    )

    return sdfg


def _make_parallel_sdfg_1(
    single_input_node: bool,
) -> tuple[dace.SDFG, dace.SDFGState]:
    """Make a parallel SDFG.

    The maps access both the same Data but uses different AccessNodes for that.
    If `single_input_node` is `True` then there will only one AccessNode for `a`
    be created, otherwise each map has its own.
    """
    sdfg = dace.SDFG(util.unique_name("parallel_sdfg_1"))
    state = sdfg.add_state(is_start_block=True)

    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10,),
            dtype=dace.float64,
            transient=False,
        )

    a1, b, c = (state.add_access(name) for name in "abc")
    a2 = a1 if single_input_node else state.add_access("a")

    state.add_mapped_tasklet(
        "map1",
        map_ranges={"__i0": "0:10"},
        inputs={"__in": dace.Memlet("a[__i0]")},
        code="__out = __in + 10.",
        outputs={"__out": dace.Memlet("b[__i0]")},
        input_nodes={a1},
        output_nodes={b},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "map2",
        map_ranges={"__i1": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1]")},
        code="__out = __in - 10.",
        outputs={"__out": dace.Memlet("c[__i1]")},
        input_nodes={a2},
        output_nodes={c},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state


def _make_multi_producer_intermediate() -> tuple[dace.SDFG, dace.SDFGState]:
    """Produces an SDFG where the intermediate node has multiple producer.

    It can be fused because the downstream map only updates it partially.
    """
    sdfg = dace.SDFG(util.unique_name("multi_producer_sdfg"))
    state = sdfg.add_state(is_start_block=True)
    state2 = sdfg.add_state_after(state)

    anames = ["a", "t", "o1", "o2"]
    for aname in anames:
        sdfg.add_array(
            name=aname,
            shape=(20, (15 if aname == "o1" else 20)),
            dtype=dace.float64,
            transient=(aname == "t"),
        )
    t = state.add_access("t")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i0": "0:20", "__i1": "0:5"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        output_nodes={t},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp2",
        map_ranges={"__i0": "0:20", "__i1": "15:20"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 2.0",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        output_nodes={t},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        "first_map",
        map_ranges={"__i0": "0:20", "__i1": "5:15"},
        inputs={"__in": dace.Memlet("a[__i0, __i1]")},
        code="__out = __in + 3.0",
        outputs={"__out": dace.Memlet("t[__i0, __i1]")},
        output_nodes={t},
        external_edges=True,
    )

    state.add_mapped_tasklet(
        "second_map",
        map_ranges={"__i0": "0:20", "__i1": "5:15"},
        inputs={"__in": dace.Memlet("t[__i0, __i1]")},
        code="__out = __in + 4.0",
        outputs={"__out": dace.Memlet("o1[__i0, __i1 - 5]")},
        input_nodes={t},
        external_edges=True,
    )

    state2.add_nedge(
        state2.add_access("t"),
        state2.add_access("o2"),
        dace.Memlet("t[0:20, 0:20] -> [0:20, 0:20]"),
    )
    sdfg.validate()

    return sdfg, state


def _make_parallel_inner() -> tuple[dace.SDFG, dace.SDFGState]:
    sdfg = dace.SDFG(util.unique_name("multi_producer_sdfg"))
    state = sdfg.add_state(is_start_block=True)

    for aname in "ao":
        sdfg.add_array(
            name=aname,
            shape=(20, 20, 2),
            dtype=dace.float64,
            transient=False,
        )

    o = state.add_access("o")

    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i0": "0:20", "__i1": "0:20"},
        inputs={"__in": dace.Memlet("a[__i0, __i1, 0]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("o[__i0, __i1, 0]")},
        output_nodes={o},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "comp1",
        map_ranges={"__i0": "0:20", "__i1": "0:20"},
        inputs={"__in": dace.Memlet("a[__i0, __i1, 1]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("o[__i0, __i1, 1]")},
        output_nodes={o},
        external_edges=True,
    )
    ret = sdfg.apply_transformations_repeated([dace_dataflow.MapExpansion])
    assert ret == 2
    sdfg.validate()

    return sdfg, state


def test_exclusive_itermediate():
    """Tests if the exclusive intermediate branch works."""
    N = 10
    sdfg = _make_serial_sdfg_1(N)

    # Now apply the optimizations.
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    sdfg.apply_transformations(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1
    assert "tmp" not in sdfg.arrays

    # Test if the intermediate is a scalar
    intermediate_nodes: list[dace_nodes.Node] = [
        node
        for node in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
        if node.data not in ["a", "b"]
    ]
    assert len(intermediate_nodes) == 1
    assert all(isinstance(node.desc(sdfg), dace.data.Scalar) for node in intermediate_nodes)

    a = np.random.rand(N, N)
    b = np.empty_like(a)
    ref = a + 4.0
    sdfg(a=a, b=b)

    assert np.allclose(b, ref)


def test_shared_itermediate():
    """Tests the shared intermediate path.

    The function uses the `_make_serial_sdfg_1()` SDFG. However, it promotes `tmp`
    to a global, and it thus became a shared intermediate, i.e. will survive.
    """
    N = 10
    sdfg = _make_serial_sdfg_1(N)
    sdfg.arrays["tmp"].transient = False

    # Now apply the optimizations.
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    sdfg.apply_transformations(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1
    assert "tmp" in sdfg.arrays

    # Test if the intermediate is a scalar
    intermediate_nodes: list[dace_nodes.Node] = [
        node
        for node in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
        if node.data not in ["a", "b", "tmp"]
    ]
    assert len(intermediate_nodes) == 1
    assert all(isinstance(node.desc(sdfg), dace.data.Scalar) for node in intermediate_nodes)

    a = np.random.rand(N, N)
    b = np.empty_like(a)
    tmp = np.empty_like(a)

    ref_b = a + 4.0
    ref_tmp = a + 1.0
    sdfg(a=a, b=b, tmp=tmp)

    assert np.allclose(b, ref_b)
    assert np.allclose(tmp, ref_tmp)


def test_pure_output_node():
    """Tests the path of a pure intermediate."""
    N = 10
    sdfg = _make_serial_sdfg_2(N)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 3

    # The first fusion will only bring it down to two maps.
    sdfg.apply_transformations(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    sdfg.apply_transformations(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 1

    a = np.random.rand(N, N)
    b = np.empty_like(a)
    c = np.empty_like(a)
    ref_b = a + 4.0
    ref_c = a - 4.0
    sdfg(a=a, b=b, c=c)

    assert np.allclose(b, ref_b)
    assert np.allclose(c, ref_c)


def test_array_intermediate():
    """Tests the correct working if we have more than scalar intermediate.

    The test used `_make_serial_sdfg_1()` to get an SDFG and then call `MapExpansion`.
    Map fusion is then called only outer maps, thus the intermediate node, must
    be an array.
    """
    N = 10
    sdfg = _make_serial_sdfg_1(N)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2
    sdfg.apply_transformations_repeated([dace_dataflow.MapExpansion])
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 4

    # Now perform the fusion
    sdfg.apply_transformations(
        gtx_transformations.MapFusionSerial(only_toplevel_maps=True),
        validate=True,
        validate_all=True,
    )
    map_entries = util.count_nodes(sdfg, dace_nodes.MapEntry, return_nodes=True)

    scope = next(iter(sdfg.states())).scope_dict()
    assert len(map_entries) == 3
    top_maps = [map_entry for map_entry in map_entries if scope[map_entry] is None]
    assert len(top_maps) == 1
    top_map = top_maps[0]
    assert sum(scope[map_entry] is top_map for map_entry in map_entries) == 2

    # Find the access node that is the new intermediate node.
    inner_access_nodes: list[dace_nodes.AccessNode] = [
        node
        for node in util.count_nodes(sdfg, dace_nodes.AccessNode, True)
        if scope[node] is not None
    ]
    assert len(inner_access_nodes) == 1
    inner_access_node = inner_access_nodes[0]
    inner_desc: dace.data.Data = inner_access_node.desc(sdfg)
    assert inner_desc.shape == (N,)

    a = np.random.rand(N, N)
    b = np.empty_like(a)
    ref_b = a + 4.0
    sdfg(a=a, b=b)

    assert np.allclose(ref_b, b)


def test_interstate_transient():
    """Tests if an interstate transient is handled properly.

    This function uses the SDFG generated by `_make_serial_sdfg_2()`. It adds a second
    state to SDFG in which `tmp_1` is read from and the result is written in `d` (new
    variable). Thus `tmp_1` can not be removed.
    """
    N = 10
    sdfg = _make_serial_sdfg_2(N)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 3
    assert sdfg.number_of_nodes() == 1

    # Now add the new state and the new output.
    sdfg.add_datadesc("d", copy.deepcopy(sdfg.arrays["b"]))
    head_state = next(iter(sdfg.states()))
    new_state = sdfg.add_state_after(head_state)

    new_state.add_mapped_tasklet(
        name="first_computation_second_state",
        map_ranges=[("__i0", f"0:{N}"), ("__i1", f"0:{N}")],
        inputs={"__in0": dace.Memlet("tmp_1[__i0, __i1]")},
        code="__out = __in0 + 9.0",
        outputs={"__out": dace.Memlet("d[__i0, __i1]")},
        external_edges=True,
    )

    # Now apply the transformation
    sdfg.apply_transformations_repeated(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert "tmp_1" in sdfg.arrays
    assert "tmp_2" not in sdfg.arrays
    assert sdfg.number_of_nodes() == 2
    assert util.count_nodes(head_state, dace_nodes.MapEntry) == 1
    assert util.count_nodes(new_state, dace_nodes.MapEntry) == 1

    a = np.random.rand(N, N)
    b = np.empty_like(a)
    c = np.empty_like(a)
    d = np.empty_like(a)
    ref_b = a + 4.0
    ref_c = a - 4.0
    ref_d = a + 10.0

    sdfg(a=a, b=b, c=c, d=d)
    assert np.allclose(ref_b, b)
    assert np.allclose(ref_c, c)
    assert np.allclose(ref_d, d)


def test_indirect_access():
    """Tests if indirect accesses are handled.

    Indirect accesses, a Tasklet dereferences the array, can not be fused, because
    the array is accessed by the Tasklet.
    """
    N_input = 100
    N_output = 1000
    a = np.random.rand(N_input)
    b = np.random.rand(N_input)
    c = np.empty(N_output)
    idx = np.random.randint(low=0, high=N_input, size=N_output, dtype=np.int32)
    sdfg = _make_serial_sdfg_3(N_input=N_input, N_output=N_output)
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    def _ref(a, b, idx):
        tmp = a + b
        return tmp[idx]

    ref = _ref(a, b, idx)

    sdfg(a=a, b=b, idx=idx, c=c)
    assert np.allclose(ref, c)

    # Now "apply" the transformation
    sdfg.apply_transformations_repeated(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    c[:] = -1.0
    sdfg(a=a, b=b, idx=idx, c=c)
    assert np.allclose(ref, c)


def test_indirect_access_2():
    """Indirect accesses, with non point wise input dependencies.

    Because `a` is used as input and output and `a` is indirectly accessed
    the access to `a` can not be point wise so, fusing is not possible.
    """
    sdfg = dace.SDFG(util.unique_name("indirect_access_sdfg_2"))
    state = sdfg.add_state(is_start_block=True)

    names = ["a", "b", "idx", "tmp"]

    for name in names:
        sdfg.add_array(
            name=name,
            shape=(10,),
            dtype=dace.int32 if name == "idx" else dace.float64,
            transient=False,
        )
    sdfg.arrays["tmp"].transient = True

    a_in, b, idx, tmp, a_out = (state.add_access(name) for name in (names + ["a"]))

    state.add_mapped_tasklet(
        "indirect_access",
        map_ranges={"__i0": "0:10"},
        inputs={
            "__idx": dace.Memlet("idx[__i0]"),
            "__field": dace.Memlet("a[0:10]", volume=1),
        },
        code="__out = __field[__idx]",
        outputs={"__out": dace.Memlet("tmp[__i0]")},
        input_nodes={a_in, idx},
        output_nodes={tmp},
        external_edges=True,
    )
    state.add_mapped_tasklet(
        "computation",
        map_ranges={"__i0": "0:10"},
        inputs={
            "__in1": dace.Memlet("tmp[__i0]"),
            "__in2": dace.Memlet("b[__i0]"),
        },
        code="__out = __in1 + __in2",
        outputs={"__out": dace.Memlet("a[__i0]")},
        input_nodes={tmp, b},
        output_nodes={a_out},
        external_edges=True,
    )
    sdfg.validate()

    count = sdfg.apply_transformations_repeated(
        gtx_transformations.MapFusionSerial(),
        validate=True,
        validate_all=True,
    )
    assert count == 0


def test_parallel_1():
    sdfg, state = _make_parallel_sdfg_1(single_input_node=False)
    assert util.count_nodes(state, dace_nodes.AccessNode) == 4
    assert util.count_nodes(state, dace_nodes.MapEntry) == 2

    # Because we request a common ancestor it will not apply.
    #  NOTE: We might have to change that if the implementation changes.
    nb_applies = sdfg.apply_transformations_repeated(
        [gtx_transformations.MapFusionParallel(only_if_common_ancestor=True)]
    )
    assert nb_applies == 0

    # If we do not restrict common ancestor then it will work.
    nb_applies = sdfg.apply_transformations_repeated(
        [gtx_transformations.MapFusionParallel(only_if_common_ancestor=False)]
    )

    assert nb_applies == 1
    # The access nodes are fused.
    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert util.count_nodes(state, dace_nodes.MapEntry) == 1


def test_parallel_2():
    sdfg, state = _make_parallel_sdfg_1(single_input_node=True)
    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert util.count_nodes(state, dace_nodes.MapEntry) == 2

    nb_applies = sdfg.apply_transformations_repeated([gtx_transformations.MapFusionParallel()])

    assert nb_applies == 1
    assert util.count_nodes(state, dace_nodes.AccessNode) == 3
    assert util.count_nodes(state, dace_nodes.MapEntry) == 1


def test_parallel_3():
    """Test that the parallel map fusion does not apply for serial maps."""
    sdfg = _make_serial_sdfg_1(20)
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2

    # Because the maps are fully serial, parallel map fusion should never apply.
    nb_applies = sdfg.apply_transformations_repeated(
        [gtx_transformations.MapFusionParallel(only_if_common_ancestor=False)]
    )
    assert nb_applies == 0
    assert util.count_nodes(sdfg, dace_nodes.AccessNode) == 3
    assert util.count_nodes(sdfg, dace_nodes.MapEntry) == 2


def test_multi_producer_sdfg():
    sdfg, state = _make_multi_producer_intermediate()
    assert util.count_nodes(state, dace_nodes.MapEntry) == 4

    ac_initial = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_initial) == 5
    assert any(
        state.out_degree(ac) == 1 and state.in_degree(ac) == 3
        for ac in ac_initial
        if ac.data == "t"
    )

    ref = {
        "a": np.array(np.random.rand(20, 20), dtype=np.float64, copy=True),
        "o1": np.array(np.random.rand(20, 15), dtype=np.float64, copy=True),
        "o2": np.array(np.random.rand(20, 20), dtype=np.float64, copy=True),
    }
    res = copy.deepcopy(ref)

    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applies = sdfg.apply_transformations_repeated(
        gtx_transformations.MapFusion(), validate=True, validate_all=True
    )

    sdfg.validate()
    assert nb_applies == 1

    ac_after = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_after) == 6  # The additional one is the transient.
    assert any(
        state.out_degree(ac) == 0 and state.in_degree(ac) == 3
        for ac in ac_initial
        if ac.data == "t"
    )
    assert util.count_nodes(state, dace_nodes.MapEntry) == 3

    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in ref)


def test_relocation_connectors():
    sdfg, state = _make_parallel_inner()
    assert util.count_nodes(state, dace_nodes.MapEntry) == 4

    ac_nodes_befor = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes_befor) == 3
    assert len([ac for ac in ac_nodes_befor if ac.data == "a"]) == 2

    ref = {
        "a": np.array(np.random.rand(20, 20, 2), dtype=np.float64, copy=True),
        "o": np.array(np.random.rand(20, 20, 2), dtype=np.float64, copy=True),
    }
    res = copy.deepcopy(ref)

    util.compile_and_run_sdfg(sdfg, **ref)

    nb_applies = sdfg.apply_transformations_repeated(
        gtx_transformations.MapFusion(), validate=True, validate_all=True
    )

    sdfg.validate()
    assert nb_applies == 2

    ac_nodes = util.count_nodes(state, dace_nodes.AccessNode, return_nodes=True)
    assert len(ac_nodes) == 2
    a, o = (ac_nodes[0], ac_nodes[1]) if ac_nodes[0] == "a" else (ac_nodes[1], ac_nodes[0])

    # The input data node `a` must have been reused, while the output node is never reused.
    assert state.in_degree(a) == 0 and state.out_degree(a) == 1
    assert state.in_degree(o) == 2 and state.out_degree(o) == 0

    map_entries = util.count_nodes(state, dace_nodes.MapEntry, return_nodes=True)
    assert len(map_entries) == 2
    scope_dict = state.scope_dict()
    outer_me, inner_me = (
        (map_entries[0], map_entries[1])
        if scope_dict[map_entries[0]] is None
        else (map_entries[1], map_entries[0])
    )
    # For global Maps the connectors are reused, but not for nested Maps.
    assert state.in_degree(outer_me) == 1
    assert state.out_degree(outer_me) == 2
    assert len(outer_me.out_connectors) == 1
    assert state.in_degree(inner_me) == 2
    assert state.out_degree(inner_me) == 2
    assert len(inner_me.out_connectors) == 2

    util.compile_and_run_sdfg(sdfg, **res)
    assert all(np.allclose(ref[name], res[name]) for name in res)
