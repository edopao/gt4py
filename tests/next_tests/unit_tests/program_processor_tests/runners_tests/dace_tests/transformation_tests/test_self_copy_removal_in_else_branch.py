# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest


dace = pytest.importorskip("dace")
from dace.sdfg import nodes as dace_nodes

from gt4py.next import common as gtx_common
from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)


from . import util

def test_self_copy_removal_simple():
    M = 20
    N = 256
    sdfg = dace.SDFG(gtx_transformations.utils.unique_name("self_copy_removal_simple"))
    A, _ = sdfg.add_array("A", [N], dace.float64)
    B, _ = sdfg.add_array("B", [N], dace.float64)
    mask, _ = sdfg.add_array("mask", [N], dace.int32)
    st = sdfg.add_state()

    nsdfg1 = dace.SDFG("nsdfg1")
    a1, _ = nsdfg1.add_scalar("a1", dace.float64)
    b1, _ = nsdfg1.add_scalar("b1", dace.float64)
    c1, _ = nsdfg1.add_scalar("c1", dace.float64)
    lev1, _ = nsdfg1.add_scalar("lev1", dace.int32)
    st1 = nsdfg1.add_state()

    nsdfg2 = dace.SDFG("nsdfg2")
    a2, _ = nsdfg2.add_scalar("a2", dace.float64)
    b2, _ = nsdfg2.add_scalar("b2", dace.float64)
    c2, _ = nsdfg2.add_scalar("c2", dace.float64)
    lev2, _ = nsdfg2.add_scalar("lev2", dace.int32)

    # create states inside the nested SDFG for the if-branches
    if_region = dace.sdfg.state.ConditionalBlock("if")
    nsdfg2.add_node(if_region, ensure_unique_name=True)
    entry_state = nsdfg2.add_state("entry", is_start_block=True)
    nsdfg2.add_edge(entry_state, if_region, dace.InterstateEdge())

    then_body = dace.sdfg.state.ControlFlowRegion("then_body", sdfg=nsdfg2)
    then_state = then_body.add_state("then_branch", is_start_block=True)
    if_region.add_branch(dace.sdfg.state.CodeBlock(f"{lev2} > {M/2}"), then_body)

    else_body = dace.sdfg.state.ControlFlowRegion("else_body", sdfg=nsdfg2)
    else_state = else_body.add_state("else_branch", is_start_block=True)
    # Use `None` for unconditional execution of else-branch, if the condition is not met.
    if_region.add_branch(None, else_body)

    add_tasklet = then_state.add_tasklet("add", inputs={"arg0", "arg1"}, outputs={"result"}, code="result = arg0 + arg1")
    then_state.add_edge(
        then_state.add_access(a2),
        None,
        add_tasklet,
        "arg0",
        dace.Memlet(f"{a2}[0]")
    )
    then_state.add_edge(
        then_state.add_access(b2),
        None,
        add_tasklet,
        "arg1",
        dace.Memlet(f"{b2}[0]")
    )
    then_state.add_edge(
        add_tasklet,
        "result",
        then_state.add_access(c2),
        None,
        dace.Memlet(f"{c2}[0]")
    )
    else_state.add_nedge(
        else_state.add_access(b2),
        else_state.add_access(c2),
        dace.Memlet(data=b2, subset="0", other_subset="0")
    )
    nsdfg2_node = st1.add_nested_sdfg(
        nsdfg2,
        inputs={lev2, a2, b2},
        outputs={c2},
        symbol_mapping={},
    )
    st1.add_edge(
        st1.add_access(a1),
        None,
        nsdfg2_node,
        a2,
        dace.Memlet(f"{a1}[0]")
    )
    st1.add_edge(
        st1.add_access(b1),
        None,
        nsdfg2_node,
        b2,
        dace.Memlet(f"{b1}[0]")
    )
    st1.add_edge(
        st1.add_access(lev1),
        None,
        nsdfg2_node,
        lev2,
        dace.Memlet(f"{lev1}[0]")
    )
    st1.add_edge(
        nsdfg2_node,
        c2,
        st1.add_access(c1),
        None,
        dace.Memlet(f"{c1}[0]")
    )

    nsdfg1_node = st.add_nested_sdfg(
        nsdfg1,
        inputs={lev1, a1, b1},
        outputs={c1},
        symbol_mapping={},
    )
    me, mx = st.add_map("map", dict(i=f"0:{N}"))
    st.add_memlet_path(st.add_access(A), me, nsdfg1_node, dst_conn=a1, memlet=dace.Memlet(f"{A}[i]"))
    st.add_memlet_path(st.add_access(B), me, nsdfg1_node, dst_conn=b1, memlet=dace.Memlet(f"{B}[i]"))
    st.add_memlet_path(st.add_access(mask), me, nsdfg1_node, dst_conn=lev1, memlet=dace.Memlet(f"{mask}[i]"))
    st.add_memlet_path(nsdfg1_node, mx, st.add_access(B), src_conn=c1, memlet=dace.Memlet(f"{B}[i]"))

    sdfg.validate()

    res, ref = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    if_region.remove_branch(else_body)
    # count = sdfg.apply_transformations_repeated(
    #     gtx_transformations.SingleStateGlobalSelfCopyBranchRemoval(
    #         single_use_data={sdfg: {}},
    #     ),
    #     validate=True,
    #     validate_all=True,
    # )
    # assert count == 1

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref=ref, res=res)