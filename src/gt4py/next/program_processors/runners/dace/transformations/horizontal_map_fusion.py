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
from itertools import product

def gt_horizontal_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    ret = sdfg.apply_transformations_repeated(
        [HorizontalMapFusion(), gtx_transformations.MapFusionParallel(only_if_common_ancestor=True)],
        validate=validate,
        validate_all=validate_all,
    )

    # ret +=  sdfg.apply_transformations_repeated([gtx_transformations.MapFusionParallel(only_if_common_ancestor=False)])

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

    # find overlapping range between the two maps
    def find_overlapping_range(
        self,
        range1: dace.subsets.Range,
        range2: dace.subsets.Range,
    ) -> tuple[int, int, int]:
        start = int(max(range1[0], range2[0]))
        end = int(min(range1[1], range2[1]))
        return start, end, int(range1[2])

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
        if not (len(graph.in_edges(first_map_entry)) and len(graph.in_edges(second_map_entry))):
            print("[can_be_applied] No same in edges", flush=True)
            return False

        for iedge1 in graph.in_edges(first_map_entry):
            print(f"[can_be_applied] iedge1: {iedge1} iedge1.src: {iedge1.src}", flush=True)
        for iedge2 in graph.in_edges(second_map_entry):
            print(f"[can_be_applied] iedge2: {iedge2} iedge2.src: {iedge2.src}", flush=True)

        return_value = False

        # check if there is a common connector between the two maps as input
        for iedge1 in graph.in_edges(first_map_entry):
            for iedge2 in graph.in_edges(second_map_entry):
                if iedge1.src.label == iedge2.src.label:
                    return_value = True
                    print(f"[can_be_applied] === Found common connector: {iedge1.src}", flush=True)
                    if len(first_map_entry.map.range) == len(second_map_entry.map.range):
                        all_ranges_are_the_same = True
                        for range_index in range(len(first_map_entry.map.range)):
                            range1 = first_map_entry.map.range[range_index]
                            range2 = second_map_entry.map.range[range_index]
                            print(f"[can_be_applied] range1: {range1}", flush=True)
                            print(f"[can_be_applied] range2: {range2}", flush=True)
                            overlapping_range = self.find_overlapping_range(
                                range1,
                                range2,
                            )
                            print(f"[can_be_applied] overlapping_range: {overlapping_range}", flush=True)
                            if int(overlapping_range[0]) == int(overlapping_range[1]) or int(overlapping_range[0]) > int(overlapping_range[1]):
                                print(f"[can_be_applied] range1 is not smaller than range2 at index {range_index}", flush=True)
                                return_value = False
                                break
                            if not (int(overlapping_range[0]) == int(range1[0]) and int(overlapping_range[1]) == int(range1[1]) and int(overlapping_range[0]) == int(range2[0]) and int(overlapping_range[1]) == int(range2[1])):
                                all_ranges_are_the_same = False
                            else:
                                print(f"[can_be_applied] range1 is same as range2 at index {range_index}", flush=True)
                            # TODO(iomaganaris): Stop when there is already another map that matches the range
                            # range1 = first_map_entry.map.range[range_index]
                            # range2 = second_map_entry.map.range[range_index]
                            # if range1[0] <= range2[0]:
                            #     if range1[1] >= range2[1]:
                            #         if range1[1] >= range2[0]:
                            #             print(f"[can_be_applied] === Found common connector: {iedge1.src}", flush=True)
                            #             return True
                        if all_ranges_are_the_same:
                            return_value = False

        if return_value:
            print(f"[can_be_applied] ===  True", flush=True)

        return return_value

    def apply(
        self,
        graph: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> None:
        first_map_entry: dace_nodes.MapEntry = self.first_map_entry
        first_map_exit: dace_nodes.MapExit = graph.exit_node(first_map_entry)
        second_map_entry: dace_nodes.MapEntry = self.second_map_entry
        second_map_exit: dace_nodes.MapExit = graph.exit_node(second_map_entry)

        def copy_full_map(
            graph: dace.SDFGState,
            map_entry: dace_nodes.MapEntry,
            map_exit: dace_nodes.MapExit,
            new_ranges: list[dace.subsets.Range],
        ) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:            
            new_nodes = {}

            map_nodes = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True).nodes()
            map_edges = graph.scope_subgraph(map_entry, include_entry=True, include_exit=True).edges()

            map_entry_copy = None
            map_exit_copy = None

            for node in map_nodes:
                node_ = copy.deepcopy(node)
                graph.add_node(node_)
                new_nodes[node] = node_
                if node == map_entry:
                    node_.map.label = f"{node.label}_copy"
                    for range_index in range(len(map_entry.map.range)):
                        node_.map.range[range_index] = new_ranges[range_index]
                    map_entry_copy = node_
                elif node == map_exit:
                    node_.map.label = f"{node.label}_copy"
                    for range_index in range(len(map_exit.map.range)):
                        node_.map.range[range_index] = new_ranges[range_index]
                    map_exit_copy = node_

            for edge in map_edges:
                print(f"[apply] map_edges edge: {edge}", flush=True)
                if edge.src in map_nodes and edge.dst in map_nodes and edge.src != map_entry and edge.dst != map_exit:
                    print("[apply] edge inside the map", flush=True)
                    graph.add_edge(new_nodes[edge.src], edge.src_conn, new_nodes[edge.dst], edge.dst_conn,
                                copy.deepcopy(edge.data))
                elif edge.dst == map_exit:
                    print("[apply] edge to map_exit", flush=True)
                    graph.add_edge(new_nodes[edge.src], edge.src_conn, map_exit_copy, edge.dst_conn, copy.deepcopy(edge.data))
                    map_exit_copy.add_in_connector(edge.dst_conn)
                elif edge.src == map_entry:
                    print("[apply] edge from map_entry", flush=True)
                    graph.add_edge(map_entry_copy, edge.src_conn, new_nodes[edge.dst], edge.dst_conn, copy.deepcopy(edge.data))
                    map_entry_copy.add_out_connector(edge.src_conn)
            
            for i, iedge in enumerate(graph.in_edges(map_entry)):
                if iedge.src_conn not in map_entry_copy.in_connectors:
                    print(f"[apply] copy map_entry iedge {i}: {iedge}", flush=True)
                    copy_memlet = copy.deepcopy(iedge.data)
                    new_volume = 1
                    for index in range(len(iedge.data.subset)):
                        print(f"[apply] iedge.data.subset: {iedge.data.subset[index]}", flush=True)
                        copy_memlet.subset[index] = new_ranges[index]
                        new_volume *= new_ranges[index][1] - new_ranges[index][0] + 1
                    copy_memlet.volume = int(new_volume)
                    graph.add_edge(iedge.src, iedge.src_conn, map_entry_copy, iedge.dst_conn, copy_memlet)

            for i, oedge in enumerate(graph.out_edges(map_exit)):
                print(f"[apply] copy map_exit oedge {i}: {oedge}", flush=True)
                copy_memlet = copy.deepcopy(oedge.data)
                new_volume = 1
                for index in range(len(oedge.data.subset)):
                    print(f"[apply] oedge.data.subset: {oedge.data.subset[index]}", flush=True)
                    copy_memlet.subset[index] = new_ranges[index]
                    new_volume *= new_ranges[index][1] - new_ranges[index][0] + 1
                copy_memlet.volume = int(new_volume)
                graph.add_edge(map_exit_copy, oedge.src_conn, oedge.dst, oedge.dst_conn, copy_memlet)

            return map_entry_copy, map_exit_copy
        
        overlapping_ranges = []
        for range_index in range(len(first_map_entry.map.range)):
            overlapping_ranges.append(self.find_overlapping_range(
                first_map_entry.map.range[range_index],
                second_map_entry.map.range[range_index],
            ))
        
        copy_full_map(graph, first_map_entry, first_map_exit, overlapping_ranges)

        # write a function that finds ranges that are not overlapping. for example range1: 0:10 and range2: 5:15 should be set to 0:5 and 10:15
        def find_non_overlapping_ranges(
            range1: dace.subsets.Range,
            range2: dace.subsets.Range,
        ) -> list[tuple[int, int, int]]:
            start1, end1, stride1 = range1
            start2, end2, stride2 = range2
            assert stride1 == stride2, "Strides must be the same"
            ranges = []
            if start1 < start2:
                ranges.append((int(start1), int(start2-1), int(stride1)))
            if end1 > end2:
                ranges.append((int(end2+1), int(end1), int(stride1)))
            if len(ranges) == 0:
                ranges.append((int(start1), int(end1), int(stride1)))
            return ranges
        
        # for every range in first_map_entry.map.range create all the combinations of non overlapping ranges
        # and add them to a list
        print(f"[apply] first_map_entry.map.range: {first_map_entry.map.range}", flush=True)
        print(f"[apply] second_map_entry.map.range: {second_map_entry.map.range}", flush=True)
        ranges_to_add = []
        for range_index in range(len(first_map_entry.map.range)):
            print(f"[apply] range_index: {range_index}", flush=True)
            print(f"[apply] first_map_entry.map.range[range_index]: {first_map_entry.map.range[range_index]}", flush=True)
            print(f"[apply] overlapping_ranges[range_index]: {overlapping_ranges[range_index]}", flush=True)
            ranges_to_add.append(find_non_overlapping_ranges(
                first_map_entry.map.range[range_index],
                overlapping_ranges[range_index],
            ))
        print(f"[apply] ranges_to_add: {ranges_to_add}", flush=True)
        # Generate all possible combinations of ranges_to_add
        range_combinations = list(product(*ranges_to_add))
        print(f"[apply] range_combinations: {range_combinations}", flush=True)

        # Iterate through each combination and create copies of the map
        for combination in range_combinations:
            print(f"[apply] Processing combination: {combination}", flush=True)
            copy_full_map(graph, first_map_entry, first_map_exit, list(combination))

        for node in graph.scope_subgraph(first_map_entry, include_entry=True, include_exit=True).nodes():
            print(f"[apply] first_map_entry node: {node}", flush=True)
            graph.remove_node(node)

        # sdfg.view()
        # import pdb; pdb.set_trace()  # noqa: E701
