# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of scan field operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module. This module implements the scan
translator, separately from the `gtir_to_sdfg_primitives` module, because the
parsing of input arguments as well as the construction of the map scope differ
from a regular field operator, which requires slightly different helper methods.
Besides, the function code is quite large, another reason to keep it separate
from other translators.

The current GTIR representation of the scan operator is based on iterator view.
This is likely to change in the future, to enable GTIR optimizations for scan.
"""

from __future__ import annotations

from typing import Sequence

import dace
from dace import subsets as dace_subsets

from gt4py import eve
from gt4py.eve.extended_typing import MaybeNestedInTuple
from gt4py.next import utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import infer_domain
from gt4py.next.program_processors.runners.dace.lowering import (
    gtir_dataflow,
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_info as ti, type_specifications as ts


def _parse_scan_fieldop_arg(
    node: gtir.Expr,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
) -> MaybeNestedInTuple[gtir_dataflow.MemletExpr]:
    """Helper method to visit an expression passed as argument to a scan field operator.

    On the innermost level, a scan operator is lowered to a loop region which computes
    column elements in the vertical dimension.

    It differs from the helper method `gtir_to_sdfg_primitives` in that field arguments
    are passed in full shape along the vertical dimension, rather than as iterator.
    """

    def _parse_fieldop_arg_impl(
        arg: gtir_to_sdfg_types.FieldopData,
    ) -> gtir_dataflow.MemletExpr:
        arg_expr = arg.get_local_view(field_domain, ctx.sdfg)
        if isinstance(arg_expr, gtir_dataflow.MemletExpr):
            return arg_expr
        # In scan field operator, the arguments to the vertical stencil are passed by value.
        # Therefore, the full field shape is passed as `MemletExpr` rather than `IteratorExpr`.
        field_type = ts.FieldType(
            dims=[dim for dim, _ in arg_expr.field_domain], dtype=arg_expr.gt_dtype
        )
        return gtir_dataflow.MemletExpr(
            arg_expr.field, field_type, arg_expr.get_memlet_subset(ctx.sdfg)
        )

    arg = sdfg_builder.visit(node, ctx=ctx)

    if isinstance(arg, gtir_to_sdfg_types.FieldopData):
        return _parse_fieldop_arg_impl(arg)
    else:
        # handle tuples of fields
        return gtx_utils.tree_map(_parse_fieldop_arg_impl)(arg)


def _scan_input_name(input_name: str) -> str:
    """
    Helper function to make naming of input connectors in the scan nested SDFG
    consistent throughut this module scope.
    """
    return f"gtir_scan_input_{input_name}"


def _scan_output_name(input_name: str) -> str:
    """
    Same as above, but for the output connecters in the scan nested SDFG.
    """
    return f"gtir_scan_output_{input_name}"


def _write_scan_output(
    ctx: gtir_to_sdfg.SubgraphContext,
    scan_carry_sym: gtir.Sym,
    output_edge: gtir_dataflow.DataflowOutputEdge,
    field_domain: gtir_domain.FieldopDomain,
    map_exit: dace.nodes.MapExit,
) -> gtir_to_sdfg_types.FieldopResult:
    # the memory layout of the output field follows the field operator compute domain
    field_dims, field_origin, _ = gtir_domain.get_field_layout(field_domain)
    field_indices = gtir_domain.get_domain_indices(field_dims, field_origin)
    field_subset = dace_subsets.Range.from_indices(field_indices)

    carry_name = str(scan_carry_sym.id)
    field_name = _scan_output_name(carry_name)
    field_node = ctx.state.add_access(field_name)

    # and here the edge writing the dataflow result data through the map exit node
    output_edge.connect(map_exit, field_node, field_subset)

    return gtir_to_sdfg_types.FieldopData(
        field_node, ts.FieldType(field_dims, output_edge.result.gt_dtype), tuple(field_origin)
    )


def _lower_lambda_to_nested_sdfg(
    lambda_node: gtir.Lambda,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    field_domain: gtir_domain.FieldopDomain,
    lambda_params: Sequence[gtir.Sym],
    scan_forward: bool,
    node_type: ts.FieldType | ts.TupleType,
) -> tuple[gtir_to_sdfg.SubgraphContext, gtir_to_sdfg_types.FieldopResult]:
    """
    Helper method to lower the lambda node representing the scan stencil dataflow
    inside a separate SDFG.

    In regular field operators, where the computation of a grid point is independent
    from other points, therefore the stencil can be lowered to a mapped tasklet
    dataflow, and the map range is defined on the full domain.
    The scan field operator has to carry an intermediate result while the stencil
    is applied on vertical levels, which is input to the computation of next level
    (an accumulator function, for example). Therefore, the points on the vertical
    dimension are computed inside a `LoopRegion` construct.
    This function creates the `LoopRegion` inside a nested SDFG, which will be
    mapped by the caller to the horizontal domain in the field operator context.

    Args:
        lambda_node: The lambda representing the stencil expression on the scan dimension.
        ctx: The SDFG context where the scan field operator is translated.
        sdfg_builder: The SDFG builder object to access the field operator context.
        field_domain: The field operator domain, with all horizontal and vertical dimensions.
        lambda_params: List of symbols used as parameters of the lambda expression.
        scan_forward: When True, the loop should range starting from the origin;
            when False, traverse towards origin.
        scan_carry_symbol: The symbol used in the stencil expression to carry the
            intermediate result along the vertical dimension.
        node_type:

    Returns:
        A tuple of two elements:
          - The subgraph context containing the `LoopRegion` along the vertical
            dimension, to be instantied as a nested SDFG in the field operator context.
          - The inner fields, that is 1d arrays with vertical shape containing
            the output of the stencil computation. These fields will have to be
            mapped to outer arrays by the caller. The caller is responsible to ensure
            that inner and outer arrays use the same strides.
    """

    # We pass an empty set as symbolic arguments, which implies that all scalar
    # inputs of the scan nested SDFG will be represented as scalar data containers.
    # The reason why we do not check for dace symbolic expressions and do not map
    # them to inner symbols is that the scan expression should not contain any domain
    # expression (no field operator inside).
    assert not any(
        eve.walk_values(lambda_node).map(
            lambda node: cpm.is_call_to(node, ("cartesian_domain", "unstructured_domain"))
        )
    )

    # Create a loop region over the vertical dimension corresponding to the scan column
    column_range = next(r for r in field_domain if sdfg_builder.is_column_axis(r.dim))
    scan_loop_var = gtir_to_sdfg_utils.get_map_variable(column_range.dim)
    if scan_forward:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} < {column_range.stop}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {column_range.start}",
            update_expr=f"{scan_loop_var} = {scan_loop_var} + 1",
            inverted=False,
        )
        prev_level_offset = -1
    else:
        scan_loop = dace.sdfg.state.LoopRegion(
            label="scan",
            condition_expr=f"{scan_loop_var} >= {column_range.start}",
            loop_var=scan_loop_var,
            initialize_expr=f"{scan_loop_var} = {column_range.stop} - 1",
            update_expr=f"{scan_loop_var} = {scan_loop_var} - 1",
            inverted=False,
        )
        prev_level_offset = 1

    first_level = "gtir_scan_first_level"
    first_level_sym = im.sym(first_level, ts.ScalarType(ts.ScalarKind.BOOL))
    scan_carry_symbol = lambda_params[0]
    scan_carry_name = str(scan_carry_symbol.id)
    scan_input_sym = im.sym(_scan_input_name(scan_carry_name), scan_carry_symbol.type)
    scan_output_sym = im.sym(_scan_output_name(scan_carry_name), node_type)

    # the lambda expression, i.e. body of the scan, will be created inside a nested SDFG.
    scan_params = [*lambda_params[1:], first_level_sym, scan_input_sym, scan_output_sym]
    scan_ctx = sdfg_builder.setup_nested_context(
        lambda_node,
        "scan",
        ctx,
        scan_params,
        symbolic_inputs={first_level},
        capture_scope_symbols=False,
    )

    # We set `using_explicit_control_flow=True` because the vertical scan is lowered to a `LoopRegion`.
    # This property is used by pattern matching in SDFG transformation framework
    # to skip those transformations that do not yet support control flow blocks.
    scan_ctx.sdfg.using_explicit_control_flow = True

    scan_ctx.sdfg.add_node(scan_loop, ensure_unique_name=True)
    scan_ctx.sdfg.add_edge(
        scan_ctx.state, scan_loop, dace.InterstateEdge(assignments={first_level: True})
    )

    compute_state = scan_loop.add_state("scan_compute")
    update_state = scan_loop.add_state("scan_update")
    scan_loop.add_edge(
        compute_state, update_state, dace.InterstateEdge(assignments={first_level: False})
    )

    ext_lambda_node = im.lambda_(*scan_params)(
        im.let(
            scan_carry_symbol,
            im.if_(
                im.ref(first_level_sym.id),
                im.deref(scan_input_sym.id),
                im.deref(scan_output_sym.id),
            ),
        )(lambda_node.expr)
    )
    # annotate the 'if_' node
    ext_lambda_node.expr.args[0].type = scan_carry_symbol.type
    # annotate the 'if_' condition, that is the 'eq' node
    ext_lambda_node.expr.args[0].args[0].type = ts.ScalarType(ts.ScalarKind.BOOL)

    # inside the 'compute' state, visit the list of arguments to be passed to the stencil
    compute_ctx = gtir_to_sdfg.SubgraphContext(scan_ctx.sdfg, compute_state, scan_ctx.scope_symbols)
    stencil_args = [
        _parse_scan_fieldop_arg(im.ref(p.id), compute_ctx, sdfg_builder, field_domain)
        for p in scan_params
    ]
    scan_output_arg_idx = scan_params.index(scan_output_sym)
    stencil_args[scan_output_arg_idx] = gtx_utils.tree_map(
        lambda x: x.shift(column_range.dim, prev_level_offset)
    )(stencil_args[scan_output_arg_idx])
    # stil inside the 'compute' state, generate the dataflow representing the stencil
    # to be applied on the horizontal domain
    dataflow_input_edges, dataflow_output = gtir_dataflow.translate_lambda_to_dataflow(
        compute_ctx.sdfg, compute_ctx.state, sdfg_builder, ext_lambda_node, stencil_args
    )

    if len(field_domain) == 1:
        assert sdfg_builder.is_column_axis(field_domain[0].dim)
        # create a trivial map for zero-dimensional fields
        map_range = {
            "__gt4py_zerodim": "0",
        }
    else:
        # create map range corresponding to the field operator domain
        map_range = {
            gtir_to_sdfg_utils.get_map_variable(r.dim): f"{r.start}:{r.stop}"
            for r in field_domain
            if not sdfg_builder.is_column_axis(r.dim)
        }
    map_entry, map_exit = sdfg_builder.add_map("scan_fieldop", compute_state, map_range)

    # here we setup the edges passing through the map entry node
    for edge in dataflow_input_edges:
        edge.connect(map_entry)

    scan_carry_tree = (
        gtir_to_sdfg_utils.make_symbol_tree(scan_carry_symbol.id, scan_carry_symbol.type)
        if isinstance(scan_carry_symbol.type, ts.TupleType)
        else scan_carry_symbol
    )

    scan_output = gtx_utils.tree_map(
        lambda scan_carry_sym_,
        dataflow_output_,
        ctx=compute_ctx,
        field_domain=field_domain,
        map_exit=map_exit: _write_scan_output(
            ctx,
            scan_carry_sym_,
            dataflow_output_,
            field_domain,
            map_exit,
        )
    )(scan_carry_tree, dataflow_output)

    return scan_ctx, scan_output


def _handle_dataflow_result_of_nested_sdfg(
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    nsdfg_node: dace.nodes.NestedSDFG,
    inner_ctx: gtir_to_sdfg.SubgraphContext,
    outer_ctx: gtir_to_sdfg.SubgraphContext,
    inner_data: gtir_to_sdfg_types.FieldopData,
    outer_data: gtir_to_sdfg_types.FieldopData,
    result_domain: infer_domain.NonTupleDomainAccess,
) -> gtir_to_sdfg_types.FieldopData | None:
    assert isinstance(inner_data.gt_type, ts.FieldType)
    inner_dataname = inner_data.dc_node.data
    inner_desc = inner_data.dc_node.desc(inner_ctx.sdfg)
    assert not inner_desc.transient  # this field should aready be an inout array

    if isinstance(result_domain, domain_utils.SymbolicDomain):
        # The field is used outside the nested SDFG, therefore it needs to be copied
        # to a temporary array in the parent SDFG (outer context).
        outer_data_name = outer_data.dc_node.data
        outer_node = outer_ctx.state.add_access(outer_data_name)
        outer_ctx.state.add_edge(
            nsdfg_node,
            inner_dataname,
            outer_node,
            None,
            outer_ctx.sdfg.make_array_memlet(outer_data_name),
        )
        return gtir_to_sdfg_types.FieldopData(outer_node, outer_data.gt_type, outer_data.origin)
    else:
        # The field is not used outside the nested SDFG. It is likely just storage
        # for some internal state, accessed during column scan, and can be turned
        # into a transient array inside the nested SDFG.
        assert result_domain == infer_domain.DomainAccessDescriptor.NEVER
        nsdfg_node.out_connectors.pop(inner_dataname)
        assert len(list(outer_ctx.state.in_edges_by_connector(nsdfg_node, inner_dataname))) == 1
        in_edge = next(outer_ctx.state.in_edges_by_connector(nsdfg_node, inner_dataname))
        assert isinstance(in_edge.src, dace.nodes.AccessNode)
        outer_ctx.state.remove_edge(in_edge)
        assert outer_ctx.state.degree(in_edge.src) == 0
        outer_ctx.state.remove_node(in_edge.src)
        outer_ctx.sdfg.remove_data(in_edge.src.data)
        nsdfg_node.in_connectors.pop(inner_dataname)
        inner_desc.transient = True
        # now shrink the inner array, by removing the column dimension
        column_dim = next(d for d in inner_data.gt_type.dims if sdfg_builder.is_column_axis(d))
        column_dim_index = inner_data.gt_type.dims.index(column_dim)
        new_shape = [size for i, size in enumerate(inner_desc.shape) if i != column_dim_index]
        new_strides = [
            stride for i, stride in enumerate(inner_desc.strides) if i != column_dim_index
        ]
        inner_desc.set_shape(new_shape, new_strides)
        # update all read/write memlets
        for st in inner_ctx.sdfg.states():
            for node in st.data_nodes():
                if node.data == inner_dataname:
                    for edge in st.in_edges(node):
                        edge.data.get_dst_subset(edge, st).pop([column_dim_index])
                        assert isinstance(edge.src, dace.nodes.MapExit)
                        map_in_connector = edge.src_conn.replace("OUT_", "IN_")
                        map_in_edge = next(st.in_edges_by_connector(edge.src, map_in_connector))
                        map_in_edge.data.get_dst_subset(map_in_edge, st).pop([column_dim_index])
                    for edge in st.out_edges(node):
                        edge.data.get_src_subset(edge, st).pop([column_dim_index])
                        assert isinstance(edge.dst, dace.nodes.MapEntry)
                        map_out_connector = edge.dst_conn.replace("IN_", "OUT_")
                        map_out_edge = next(st.out_edges_by_connector(edge.dst, map_out_connector))
                        map_out_edge.data.get_src_subset(map_out_edge, st).pop([column_dim_index])
        return None


def _make_scan_compute_outer_arg(
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    ctx: gtir_to_sdfg.SubgraphContext,
    inner_ctx: gtir_to_sdfg.SubgraphContext,
    inner_data: gtir_to_sdfg_types.FieldopData,
    field_domain: gtir_domain.FieldopDomain,
) -> gtir_to_sdfg_types.FieldopData:
    _, field_origin, field_shape = gtir_domain.get_field_layout(field_domain)
    outer, _ = sdfg_builder.add_temp_array(
        ctx.sdfg, field_shape, inner_data.dc_node.desc(inner_ctx.sdfg).dtype
    )
    outer_node = ctx.state.add_access(outer)
    return gtir_to_sdfg_types.FieldopData(outer_node, inner_data.gt_type, tuple(field_origin))


def translate_scan(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Generates the dataflow subgraph for the `as_fieldop` builtin with a scan operator.

    It differs from `translate_as_fieldop()` in that the horizontal domain is lowered
    to a map scope, while the scan column computation is lowered to a `LoopRegion`
    on the vertical dimension, that is inside the horizontal map.
    The current design choice is to keep the map scope on the outer level, and
    the `LoopRegion` inside. This choice follows the GTIR representation where
    the `scan` operator is called inside the `as_fieldop` node.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert isinstance(node, gtir.FunCall)
    assert cpm.is_call_to(node.fun, "as_fieldop")
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    fun_node = node.fun
    assert len(fun_node.args) == 2
    scan_expr, scan_domain_expr = fun_node.args
    assert cpm.is_call_to(scan_expr, "scan")

    # parse the domain of the scan field operator
    assert isinstance(scan_domain_expr.type, ts.DomainType)
    field_domain = gtir_domain.get_field_domain(
        domain_utils.SymbolicDomain.from_expr(scan_domain_expr)
    )

    # parse scan parameters
    assert len(scan_expr.args) == 3
    stencil_expr = scan_expr.args[0]
    assert isinstance(stencil_expr, gtir.Lambda)

    # params[0]: the lambda parameter to propagate the scan carry on the vertical dimension
    scan_carry = stencil_expr.params[0].id
    scan_carry_type = stencil_expr.params[0].type
    assert isinstance(scan_carry_type, ts.DataType)

    # params[1]: boolean flag for forward/backward scan
    assert isinstance(scan_expr.args[1], gtir.Literal) and ti.is_logical(scan_expr.args[1].type)
    scan_forward = scan_expr.args[1].value == "True"

    # params[2]: the expression that computes the value for scan initialization
    init_expr = scan_expr.args[2]
    # visit the initialization value of the scan expression
    init_data = sdfg_builder.visit(init_expr, ctx=ctx)

    # define the symbols passed as parameter to the lambda expression, which consists
    # of the carry argument and all lambda function arguments
    lambda_arg_types: list[ts.DataType] = [scan_carry_type] + [
        arg.type for arg in node.args if isinstance(arg.type, ts.DataType)
    ]
    lambda_params = [
        im.sym(p.id, arg_type)
        for p, arg_type in zip(stencil_expr.params, lambda_arg_types, strict=True)
    ]

    # lower the scan stencil expression in a separate SDFG context
    lambda_ctx, lambda_output = _lower_lambda_to_nested_sdfg(
        stencil_expr,
        ctx,
        sdfg_builder,
        field_domain,
        lambda_params,
        scan_forward,
        node.type,
    )

    # visit the arguments to be passed to the lambda expression
    # this must be executed before visiting the lambda expression, in order to populate
    # the data descriptor with the correct field domain offsets for field arguments
    lambda_args = [sdfg_builder.visit(arg, ctx=ctx) for arg in node.args]
    outer_scan_compute_arg = gtx_utils.tree_map(
        lambda x: _make_scan_compute_outer_arg(sdfg_builder, ctx, lambda_ctx, x, field_domain)
    )(lambda_output)
    lambda_args_mapping = [
        (im.sym(_scan_input_name(scan_carry), scan_carry_type), init_data),
        (im.sym(_scan_output_name(scan_carry), node.type), outer_scan_compute_arg),
    ] + [
        (gt_symbol, arg)
        for gt_symbol, arg in zip(stencil_expr.params[1:], lambda_args, strict=True)
    ]

    lambda_arg_nodes, symbolic_args = gtir_to_sdfg.flatten_tuple_args(lambda_args_mapping)

    # The lambda expression of a scan field operator should never capture symbols
    # from the ouside scope, therefore we call `add_nested_sdfg()` with `capture_outer_data=False`.
    nsdfg_node, input_memlets = sdfg_builder.add_nested_sdfg(
        node=stencil_expr,
        inner_ctx=lambda_ctx,
        outer_ctx=ctx,
        symbolic_args=symbolic_args,
        data_args=lambda_arg_nodes,
        inner_result=lambda_output,
        capture_outer_data=False,
    )

    for input_connector, memlet in input_memlets.items():
        src = lambda_arg_nodes[input_connector]
        assert src is not None
        ctx.state.add_edge(src.dc_node, None, nsdfg_node, input_connector, memlet)

    # for output connections, we create temporary arrays that contain the computation
    # results of a column slice for each point in the horizontal domain
    return gtx_utils.tree_map(
        lambda inner_data, outer_data, output_domain: _handle_dataflow_result_of_nested_sdfg(
            sdfg_builder=sdfg_builder,
            nsdfg_node=nsdfg_node,
            inner_ctx=lambda_ctx,
            outer_ctx=ctx,
            inner_data=inner_data,
            outer_data=outer_data,
            result_domain=output_domain,
        )
    )(lambda_output, outer_scan_compute_arg, node.annex.domain)
