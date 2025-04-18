# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Union

import dace
from dace import (
    properties as dace_properties,
    subsets as dace_subsets,
    transformation as dace_transformation,
)
from dace.sdfg import nodes as dace_nodes
from dace.transformation.passes import analysis as dace_analysis

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import map_fusion_utils


def gt_vertical_map_fusion(
    sdfg: dace.SDFG,
    run_simplify: bool,
    validate: bool = True,
    validate_all: bool = False,
) -> int:
    find_single_use_data = dace_analysis.FindSingleUseData()
    single_use_data = find_single_use_data.apply_pass(sdfg, None)

    ret = sdfg.apply_transformations_repeated(
        [
            MapRangeVerticalSplit(),
            gtx_transformations.SplitAccessNode(single_use_data=single_use_data),
            gtx_transformations.MapFusionSerial(),
        ],
        validate=validate,
        validate_all=validate_all,
    )

    if run_simplify:
        gtx_transformations.gt_simplify(
            sdfg=sdfg,
            validate=validate,
            validate_all=validate_all,
        )

    return ret


@dace_properties.make_properties
class MapRangeVerticalSplit(dace_transformation.SingleStateTransformation):
    """
    Identify overlapping range between serial maps, and split the range in order
    to promote serial map fusion.
    """

    # Pattern Matching
    exit_first_map = dace_transformation.PatternNode(dace_nodes.MapExit)
    access_node = dace_transformation.PatternNode(dace_nodes.AccessNode)
    entry_second_map = dace_transformation.PatternNode(dace_nodes.MapEntry)

    @classmethod
    def expressions(cls) -> Any:
        """Get the match expressions.

        The function generates two match expressions. The first match describes
        the case where the top map must be promoted, while the second case is
        the second/lower map must be promoted.
        """
        return [
            dace.sdfg.utils.node_path_graph(
                cls.exit_first_map, cls.access_node, cls.entry_second_map
            ),
        ]

    def can_be_applied(
        self,
        graph: dace.SDFGState,
        expr_index: int,
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Check non overlapping range in the first and second map."""
        assert self.expr_index == expr_index
        splitted_range = _split_map_range(self.exit_first_map.map, self.entry_second_map.map)
        if splitted_range is None:
            return False

        first_map_splitted_range, second_map_splitted_range = splitted_range
        _, first_map_overlap_range, _ = first_map_splitted_range
        _, second_map_overlap_range, _ = second_map_splitted_range
        if (
            first_map_overlap_range == self.exit_first_map.map.range
            and second_map_overlap_range == self.entry_second_map.map.range
        ):
            return False

        return True

    def apply(self, graph: Union[dace.SDFGState, dace.SDFG], sdfg: dace.SDFG) -> None:
        """Split the map range in order to obtain an overlapping range between the first and second map."""
        splitted_range = _split_map_range(self.exit_first_map.map, self.entry_second_map.map)
        assert splitted_range is not None

        first_map_splitted_range, second_map_splitted_range = splitted_range
        first_map_lower_range, first_map_overlap_range, first_map_upper_range = (
            first_map_splitted_range
        )
        second_map_lower_range, second_map_overlap_range, second_map_upper_range = (
            second_map_splitted_range
        )

        first_map_entry: dace_nodes.MapEntry = graph.entry_node(self.exit_first_map)
        first_map_exit: dace_nodes.MapExit = self.exit_first_map
        second_map_entry: dace_nodes.MapEntry = self.entry_second_map
        second_map_exit: dace_nodes.MapExit = graph.exit_node(self.entry_second_map)

        if first_map_lower_range.num_elements() != 0:
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, first_map_entry, first_map_exit, first_map_lower_range, "lower"
            )

        if first_map_upper_range.num_elements() != 0:
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, first_map_entry, first_map_exit, first_map_upper_range, "upper"
            )

        if second_map_lower_range.num_elements() != 0:
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, second_map_entry, second_map_exit, second_map_lower_range, "lower"
            )

        if second_map_upper_range.num_elements() != 0:
            map_fusion_utils.copy_map_graph_with_new_range(
                sdfg, graph, second_map_entry, second_map_exit, second_map_upper_range, "upper"
            )

        map_fusion_utils.update_map_range(first_map_entry.map, first_map_overlap_range)
        map_fusion_utils.update_map_range(second_map_entry.map, second_map_overlap_range)


def _split_map_range(
    first_map: dace_nodes.Map,
    second_map: dace_nodes.Map,
) -> tuple[list[dace_subsets.Range], list[dace_subsets.Range]] | None:
    first_map_params = set(first_map.params)
    second_map_params = set(second_map.params)
    common_map_params = second_map_params.intersection(first_map_params)
    if len(common_map_params) == 0:
        return None
    if first_map.range.intersects(second_map.range) == False:
        # in case of disjoint ranges, we cannot find an overlapping range
        return None
    first_map_dict = dict(zip(first_map.params, first_map.range.ranges, strict=True))
    second_map_dict = dict(zip(second_map.params, second_map.range.ranges, strict=True))
    common_params = [p for p in first_map_dict.keys() if p in second_map_dict.keys()]

    first_map_common_dict = {}
    second_map_common_dict = {}
    for index in common_params:
        first_map_range = first_map_dict[index]
        second_map_range = second_map_dict[index]
        if (step := first_map_range[2]) != second_map_range[2]:
            # we do not support splitting of map range when the range step is different
            return None
        elif (first_map_range == second_map_range) == True:  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
            first_map_common_dict[index] = ((0, 0, 1), first_map_range, (0, 0, 1))
            second_map_common_dict[index] = ((0, 0, 1), second_map_range, (0, 0, 1))
        elif (first_map_range == second_map_range) == False:  # noqa: E712 [true-false-comparison]  # SymPy fuzzy bools.
            overlap_range_start = max(first_map_range[0], second_map_range[0])
            overlap_range_stop = min(first_map_range[1], second_map_range[1]) + step

            first_lower_range = [
                overlap_range_start,
                overlap_range_start - step,
                step,
            ]
            if first_map_range[0] < second_map_range[0]:
                first_lower_range[0] = first_map_range[0]
            first_upper_range = [
                overlap_range_stop,
                overlap_range_stop - step,
                step,
            ]
            if first_map_range[1] > second_map_range[1]:
                first_upper_range[1] = first_map_range[1]

            first_map_common_dict[index] = (
                first_lower_range,
                (overlap_range_start, overlap_range_stop - step, step),
                first_upper_range,
            )

            second_lower_range = [
                overlap_range_start,
                overlap_range_start - step,
                step,
            ]
            if second_map_range[0] < first_map_range[0]:
                second_lower_range[0] = second_map_range[0]
            second_upper_range = [
                overlap_range_stop,
                overlap_range_stop - step,
                step,
            ]
            if second_map_range[1] > first_map_range[1]:
                second_upper_range[1] = second_map_range[1]

            second_map_common_dict[index] = (
                second_lower_range,
                (
                    overlap_range_start,
                    overlap_range_stop - step,
                    step,
                ),
                second_upper_range,
            )
        else:
            # we cannot evaluate `first_map_range == second_map_range`, it is a symbolic expression
            return None

    first_map_splitted_range = [
        dace_subsets.Range(r)
        for r in zip(
            *(
                first_map_common_dict[p]
                if p in common_params
                else [(0, 0, 1), first_map_dict[p], (0, 0, 1)]
                for p in first_map.params
            )
        )
    ]
    second_map_splitted_range = [
        dace_subsets.Range(r)
        for r in zip(
            *(
                second_map_common_dict[p]
                if p in common_params
                else [(0, 0, 1), second_map_dict[p], (0, 0, 1)]
                for p in second_map.params
            )
        )
    ]

    return first_map_splitted_range, second_map_splitted_range
