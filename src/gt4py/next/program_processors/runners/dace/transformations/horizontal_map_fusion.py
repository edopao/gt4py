# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from typing import Any, Mapping, Optional

import dace
from dace import properties as dace_properties, transformation as dace_transformation
from dace.sdfg import nodes as dace_nodes, graph
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

def gt_horizontal_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    ret = sdfg.apply_transformations( #_once_everywhere(
        HorizontalMapFusion(),
        validate=validate,
        validate_all=validate_all,
    )

    ret +=  sdfg.apply_transformations_repeated([gtx_transformations.MapFusionParallel(only_if_common_ancestor=False)])

    if run_simplify:
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
        )

    return ret

@dace_properties.make_properties
class HorizontalMapFusion(dace_transformation.SingleStateTransformation):
    """Fuses two maps that are adjacent to each other in the same state.
    """

    first_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)
    second_map_entry = dace_transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        map_fusion_parallel_match = graph.OrderedMultiDiConnectorGraph()
        map_fusion_parallel_match.add_nodes_from(
            [cls.first_map_entry, cls.second_map_entry]
        )
        return [map_fusion_parallel_match]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry

        print(f"[can_be_applied] first_map_entry: {first_map_entry}", flush=True)
        print(f"[can_be_applied] second_map_entry: {second_map_entry}", flush=True)

        # if first_map_entry == second_map_entry:
        #     return False

        # map_range: dace.subsets.Range = map_entry.map.range
        scope_dict = graph.scope_dict()

        # The map must be on the top.
        # if scope_dict[first_map_entry] is not None or scope_dict[second_map_entry] is not None:
        #     return False

        for iedge in graph.in_edges(first_map_entry):
            print(f"[can_be_applied] iedge: {iedge}", flush=True)

        # for iedge in graph.in_edges(second_map_entry):
        #     print(f"[can_be_applied] iedge: {iedge}", flush=True)

        # For ease of implementation we also require that the Map scope is only
        #  adjacent to AccessNodes.
        # if not all(
        #     isinstance(iedge.src, dace_nodes.AccessNode) for iedge in graph.in_edges(map_entry)
        # ):
        #     return False

        # map_exit: dace_nodes.MapExit = graph.exit_node(map_entry)
        # if not all(
        #     isinstance(oedge.dst, dace_nodes.AccessNode) for oedge in graph.out_edges(map_exit)
        # ):
        #     return False
        if len(graph.in_edges(first_map_entry)):
            return True

        return False

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        first_map_exit: dace_nodes.MapExit = graph.exit_node(first_map_entry)
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        second_map_exit: dace_nodes.MapExit = graph.exit_node(second_map_entry)

        # Now find all notes that are producer or consumer of the Map, after we removed
        #  the nodes of the Maps we need to check if they have become isolated.
        for i, iedge in enumerate(graph.in_edges(first_map_entry)):
            print(f"[apply] iedge {i}: {iedge}", flush=True)

        # Create a copy of the map from first_map_entry
        first_map_entry_copy = copy.deepcopy(first_map_entry)
        first_map_exit_copy = copy.deepcopy(first_map_exit)

        first_map_entry_copy.map.label = f"{first_map_entry.label}_copy"
        first_map_exit_copy.map.label = f"{first_map_exit.label}_copy"
        first_map_entry_copy.map.range = second_map_entry.map.range
        first_map_exit_copy.map.range = second_map_exit.map.range

        # for iedge in graph.in_edges(first_map_entry_copy):
        #     graph.remove_edge_and_connectors(iedge)
        # for oedge in graph.out_edges(first_map_entry_copy):
        #     graph.remove_edge_and_connectors(oedge)
        # for iedge in graph.in_edges(first_map_exit_copy):
        #     graph.remove_edge_and_connectors(iedge)
        # for oedge in graph.out_edges(first_map_exit_copy):
        #     graph.remove_edge_and_connectors(oedge)

        # for iconn in first_map_entry.in_connectors:
        #     if iconn not in first_map_entry_copy.in_connectors:
        #         first_map_entry_copy.add_in_connector(iconn)
        # for oconn in first_map_entry.out_connectors:
        #     if oconn not in first_map_entry_copy.out_connectors:
        #         first_map_entry_copy.add_out_connector(oconn)

        # for iconn in first_map_exit.in_connectors:
        #     if iconn not in first_map_exit_copy.in_connectors:
        #         first_map_exit_copy.add_in_connector(iconn)
        # for oconn in first_map_exit.out_connectors:
        #     if oconn not in first_map_exit_copy.out_connectors:
        #         first_map_exit_copy.add_out_connector(oconn)

        # for iedge in graph.in_edges(first_map_entry):
        #     if isinstance(iedge.src, dace_nodes.AccessNode):
        #         # Add the edge to the second map
        #         iedge_copy = copy.deepcopy(iedge)
        #         graph.add_edge(iedge_copy.src, iedge_copy.src_conn, first_map_entry_copy, iedge_copy.dst_conn, iedge_copy.data)

        # for oedge in graph.out_edges(first_map_exit):
        #     if isinstance(oedge.dst, dace_nodes.AccessNode):
        #         # Add the edge to the first map
        #         oedge_copy = copy.deepcopy(oedge)
        #         graph.add_edge(first_map_exit_copy, oedge_copy.src_conn, oedge_copy.dst, oedge_copy.dst_conn, oedge_copy.data)

        # Add the copied map to the graph
        graph.add_node(first_map_entry_copy)
        graph.add_node(first_map_exit_copy)
        edge_to_remove = graph.add_edge(first_map_entry_copy, None, first_map_exit_copy, None, dace.memlet.Memlet())

        for iedge in graph.in_edges(first_map_entry):
            print(f"[apply] copy first_map_entry iedge {i}: {iedge}", flush=True)
            copy_memlet = copy.deepcopy(iedge.data)
            graph.add_edge(iedge.src, iedge.src_conn, first_map_entry_copy, iedge.dst_conn, copy_memlet)

        for iedge in graph.in_edges(first_map_entry_copy):
            print(f"[apply] copy iedge {i}: {iedge}", flush=True)

        for oedge in graph.out_edges(first_map_exit):
            print(f"[apply] copy first_map_exit oedge {i}: {oedge}", flush=True)
            copy_memlet = copy.deepcopy(oedge.data)
            graph.add_edge(first_map_exit_copy, oedge.src_conn, oedge.dst, oedge.dst_conn, copy_memlet)
        for oedge in graph.out_edges(first_map_exit_copy):
            print(f"[apply] copy oedge {i}: {oedge}", flush=True)

        for node in graph.scope_subgraph(first_map_entry, include_entry=False, include_exit=False).nodes():
            print(f"[apply] first_map_entry node {i}: {node}", flush=True)
            if isinstance(node, dace_nodes.Tasklet):
                new_tasklet = copy.deepcopy(node)
                new_tasklet.label = f"{node.label}_copy"
                graph.add_node(new_tasklet)
                for iedge in graph.in_edges(node):
                    if iedge.src == first_map_entry:
                        print(f"[apply] node has iedge to first_map_entry: {iedge}", flush=True)
                        copy_memlet = copy.deepcopy(iedge.data)
                        # new_tasklet.add_in_connector(iedge.src_conn)
                        graph.add_edge(first_map_entry_copy, iedge.src_conn, new_tasklet, iedge.dst_conn, copy_memlet)
                        first_map_entry_copy.add_out_connector(iedge.src_conn)
                        # graph.remove_edge(iedge)
                for oedge in graph.out_edges(node):
                    if oedge.dst == first_map_exit:
                        print(f"[apply] node has oedge to first_map_exit: {oedge}", flush=True)
                        copy_memlet = copy.deepcopy(oedge.data)
                        # new_tasklet.add_out_connector(oedge.dst_conn)
                        first_map_exit_copy.add_in_connector(oedge.dst_conn)
                        graph.add_edge(new_tasklet, oedge.src_conn, first_map_exit_copy, oedge.dst_conn, copy_memlet)
                        # graph.remove_edge(oedge)
                # for oedge in graph.out_edges(node):
                #     if oedge.dst == first_map_exit:
                #         oedge.dst = first_map_exit_copy                

        # # # Add the copied map to the graph
        # graph.add_node(first_map_entry_copy)
        # graph.add_node(first_map_exit_copy)

        # graph.add_edge(first_map_entry_copy, None, first_map_exit_copy, None, dace.memlet.Memlet())

        # Add first map inputs to the second map
        # for iedge in graph.in_edges(first_map_entry):
        #     if isinstance(iedge.src, dace_nodes.AccessNode):
        #         # Add the edge to the second map
        #         graph.add_edge(iedge.src, iedge.src_conn, second_map_entry, iedge.dst_conn, iedge.data)
        #         second_map_entry
        # for iconnector in first_map_entry.in_connectors:
        #     if iconnector not in second_map_entry.in_connectors:
        #         second_map_entry.add_in_connector(iconnector)
        # for oconnector in first_map_entry.out_connectors:
        #     if oconnector not in second_map_entry.out_connectors:
        #         second_map_entry.add_out_connector(oconnector)
        # # Add the second map outputs to the first map
        # for oedge in graph.out_edges(second_map_exit):
        #     if isinstance(oedge.dst, dace_nodes.AccessNode):
        #         # Add the edge to the first map
        #         graph.add_edge(second_map_exit, oedge.src_conn, oedge.dst, oedge.dst_conn, oedge.data)
        # for iconnector in first_map_exit.in_connectors:
        #     if iconnector not in second_map_exit.in_connectors:
        #         second_map_exit.add_in_connector(iconnector)
        # for oconnector in first_map_exit.out_connectors:
        #     if oconnector not in second_map_exit.out_connectors:
        #         second_map_exit.add_out_connector(oconnector)

        # Add a tasklet to the new map, similar to the one in first_map_entry
        # for tasklet in graph.scope_subgraph(first_map_entry, include_entry=False, include_exit=False).nodes():
        #     if isinstance(tasklet, dace_nodes.Tasklet):
        #         new_tasklet = copy.deepcopy(tasklet)
        #         new_tasklet.label = f"{tasklet.label}_copy"
        #         for iedge in graph.in_edges(tasklet):
        #             graph.remove_edge_and_connectors(iedge)
        #         for oedge in graph.out_edges(tasklet):
        #             graph.remove_edge_and_connectors(oedge)
        #         # Add the new tasklet to the graph
        #         graph.add_node(new_tasklet)
        #         # Connect the new tasklet to the new map entry and exit
        #         for iedge in graph.in_edges(tasklet):
        #             graph.add_edge(iedge.src, iedge.src_conn, new_tasklet, iedge.dst_conn, iedge.data)
        #         for oedge in graph.out_edges(tasklet):
        #             graph.add_edge(new_tasklet, oedge.src_conn, second_map_exit, oedge.dst_conn, oedge.data)

        # Remove the edge from new_map_entry to new_map_exit_copy
        # Fixes issue with cycle
        # edges_to_remove = graph.edges_between(
        #     second_map_entry, second_map_exit)
        # if edges_to_remove:
        #     for edge in edges_to_remove:
        graph.remove_edge(edge_to_remove)

        # Modify the range of the original map to exclude the first_map_copy range
        for i, range in enumerate(first_map_entry.map.range):
            print("[apply] range: ", range, flush=True)
            print("[apply] first_map_entry_copy.range: ", first_map_entry_copy.map.range[i], flush=True)
            if range[1] > first_map_entry_copy.map.range[i][1]:
                print("range[0] = ", range[0], flush=True)
                range_list = list(range)
                range_list[0] = first_map_entry_copy.map.range[i][1] + 1
                first_map_entry.map.range[i] = tuple(range_list)
            print("[apply] new range: ", range, flush=True)
