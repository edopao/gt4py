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
from dace.sdfg import nodes as dace_nodes


def copy_full_map(
    sdfg: dace.SDFG,
    graph: dace.SDFGState,
    map_entry: dace_nodes.MapEntry,
    map_exit: dace_nodes.MapExit,
    suffix: str,
) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:            
    new_nodes = {}
    new_data_names = {}
    new_data_descriptors = {}

    subgraph = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True)
    map_nodes = subgraph.nodes()
    map_edges = subgraph.edges()

    map_entry_copy = None
    map_exit_copy = None

    for node in map_nodes:
        node_ = copy.deepcopy(node)
        if isinstance(node, dace_nodes.AccessNode):
            node_name = node_.data
            node_desc = node_.desc(graph)
            if isinstance(node_desc, dace.data.Array):
                new_name, new_data_desc = sdfg.add_array(node_name + f"_{suffix}", node_desc.shape, node_desc.dtype, node_desc.storage,
                                node_desc.location, node_desc.transient, node_desc.strides,
                                node_desc.offset, find_new_name=True)
                node_.data = new_name
                new_data_names[node_name] = new_name
                new_data_descriptors[node_name] = new_data_desc
            elif isinstance(node_desc, dace.data.Scalar):
                new_name, new_data_desc = sdfg.add_scalar(node_name + f"_{suffix}", node_desc.dtype, node_desc.storage, node_desc.transient,
                                                        node_desc.lifetime, node_desc.debuginfo, find_new_name=True)
                node_.data = new_name
                new_data_names[node_name] = new_name
                new_data_descriptors[node_name] = new_data_desc
            else:
                raise ValueError(f"Unsupported node type: {type(node_desc)}")
        elif isinstance(node, dace_nodes.NestedSDFG):
            raise NotImplementedError("Nested SDFGs are not supported yet")
        else:
            # change label to a unique name
            if not (isinstance(node, dace_nodes.MapEntry) or isinstance(node, dace_nodes.MapExit)):
                # handled later
                node_.label = f"{node.label}_{suffix}"
            else:
                node_.map.label = f"{node.label}_{suffix}"
            graph.add_node(node_)
        new_nodes[node] = node_
        if node == map_entry:
            map_entry_copy = node_
        elif node == map_exit:
            map_exit_copy = node_

    for edge in map_edges:
        print(f"[apply] map_edges edge: {edge}", flush=True)
        # import pdb; pdb.set_trace()
        copy_memlet = copy.deepcopy(edge.data)
        if edge.data.data in new_data_names:
            copy_memlet.data = new_data_names[edge.data.data]
        graph.add_edge(new_nodes[edge.src], edge.src_conn, new_nodes[edge.dst], edge.dst_conn,
                    copy_memlet)
    
    for i, iedge in enumerate(graph.in_edges(map_entry)):
        if iedge.data not in [map_entry_copy_edge.data for map_entry_copy_edge in graph.in_edges(map_entry_copy)]:
            print(f"[apply] copy map_entry iedge {i}: {iedge}", flush=True)
            copy_memlet = copy.deepcopy(iedge.data)
            graph.add_edge(iedge.src, iedge.src_conn, map_entry_copy, iedge.dst_conn, copy_memlet)

    for i, oedge in enumerate(graph.out_edges(map_exit)):
        print(f"[apply] copy map_exit oedge {i}: {oedge}", flush=True)
        copy_memlet = copy.deepcopy(oedge.data)
        graph.add_edge(map_exit_copy, oedge.src_conn, oedge.dst, oedge.dst_conn, copy_memlet)

    return map_entry_copy, map_exit_copy