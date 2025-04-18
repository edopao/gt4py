# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy

import dace
from dace import subsets as dace_subsets
from dace.sdfg import nodes as dace_nodes


def copy_map_graph(
    sdfg: dace.SDFG,
    graph: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    map_exit: dace_nodes.MapExit,
    suffix: str,
) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
    """Performs a full copy of the map graph.

    It will perform a deep copy of all the nodes between the given `map_entry`
    and `map_exit` nodes, including these two.

    Args:
        sdfg: The SDFG containing the map graph.
        graph: The SDFG state containing the map graph.
        map_entry: The entry node of the map graph.
        map_exit: The exit node of the map graph.
        suffix: String to append to the label of the copied nodes.

    Returns:
        A tuple of map entry and exit nodes, for the new map.
    """
    new_nodes = {}
    new_data_names = {}
    new_data_descriptors = {}

    subgraph = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    map_nodes = subgraph.nodes()
    map_edges = subgraph.edges()

    new_map_entry = None
    new_map_exit = None

    for node in map_nodes:
        if isinstance(node, dace_nodes.AccessNode):
            data_name = node.data
            data_desc = node.desc(sdfg)
            if isinstance(data_desc, (dace.data.Array, dace.data.Scalar)):
                new_data_desc = data_desc.clone()
                new_data_name = sdfg.add_datadesc(
                    f"{data_name}_{suffix}", new_data_desc, find_new_name=True
                )
            else:
                raise ValueError(f"Unsupported data type: {type(data_desc)}")
            node_ = graph.add_access(new_data_name)
            new_data_names[data_name] = new_data_name
            new_data_descriptors[data_name] = new_data_desc
        elif isinstance(node, dace_nodes.NestedSDFG):
            node_ = graph.add_nested_sdfg(
                node.sdfg, sdfg, node.in_connectors, node.out_connectors, node.symbol_mapping
            )
        else:
            node_ = copy.deepcopy(node)
            # change label to a unique name
            if isinstance(node, (dace_nodes.MapEntry, dace_nodes.MapExit)):
                node_.map.label = f"{node.label}_{suffix}"
            else:
                node_.label = f"{node.label}_{suffix}"
            graph.add_node(node_)

        new_nodes[node] = node_
        if node == map_entry:
            new_map_entry = node_
        elif node == map_exit:
            new_map_exit = node_

    # we have to ensure that the exit node references the new map node
    new_map_exit.map = new_map_entry.map

    for edge in map_edges:
        copy_memlet = copy.deepcopy(edge.data)
        if edge.data.data in new_data_names:
            copy_memlet.data = new_data_names[edge.data.data]
        graph.add_edge(
            new_nodes[edge.src], edge.src_conn, new_nodes[edge.dst], edge.dst_conn, copy_memlet
        )

    for iedge in graph.in_edges(map_entry):
        copy_memlet = copy.deepcopy(iedge.data)
        graph.add_edge(iedge.src, iedge.src_conn, new_map_entry, iedge.dst_conn, copy_memlet)

    for oedge in graph.out_edges(map_exit):
        copy_memlet = copy.deepcopy(oedge.data)
        graph.add_edge(new_map_exit, oedge.src_conn, oedge.dst, oedge.dst_conn, copy_memlet)

    return new_map_entry, new_map_exit


def update_map_range(map: dace_nodes.Map, new_range: dace_subsets.Range) -> None:
    """Helper function to modify the range of a map.

    In a map graph the range is referenced in multiple nodes: the map entry,
    the exit node and the map itself. Therefore, we update the content of the list.

    Args:
        map: The map to modify.
        new_range: The range to set on the map.
    """
    for i, r in enumerate(new_range):
        map.range[i] = r


def copy_map_graph_with_new_range(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    map_exit: dace_nodes.MapExit,
    map_range: dace_subsets.Range,
    suffix: str,
) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
    """Wrapper of `copy_map_graph` that additionally calls `update_map_range` on the new map.

    It will perform a full copy of the map scope, as described in `copy_map_graph`
    and additionally set the given range on the new map.

    Args:
        sdfg: The SDFG containing the map graph.
        graph: The SDFG state containing the map graph.
        map_entry: The entry node of the map graph.
        map_exit: The exit node of the map graph.
        map_range: The range to set on the new map.
        suffix: String to append to the label of the copied nodes.

    Returns:
        A tuple of map entry and exit nodes, for the new map.
    """
    new_map_entry, new_map_exit = copy_map_graph(sdfg, state, map_entry, map_exit, suffix)
    update_map_range(new_map_entry.map, map_range)
    assert new_map_entry.range == map_range
    assert new_map_exit.map.range == map_range
    return new_map_entry, new_map_exit
