# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""Implements Helper functionaliyies for map fusion

THIS FILE WAS COPIED FROM DACE TO FACILITATE DEVELOPMENT UNTIL THE PR#1625 IN
DACE IS MERGED AND THE VERSION WAS UPGRADED.
"""


# ruff: noqa

import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union, Callable, TypeAlias

import dace
from dace import data, properties, subsets, symbolic, transformation
from dace.sdfg import SDFG, SDFGState, nodes, validation
from dace.transformation import helpers

FusionCallback: TypeAlias = Callable[
    ["MapFusionHelper", nodes.MapEntry, nodes.MapEntry, dace.SDFGState, dace.SDFG, bool], bool
]
"""Callback for the map fusion transformation to check if a fusion should be performed.
"""


@properties.make_properties
class MapFusionHelper(transformation.SingleStateTransformation):
    """Common parts of the parallel and serial map fusion transformation.

    Args:
        only_inner_maps: Only match Maps that are internal, i.e. inside another Map.
        only_toplevel_maps: Only consider Maps that are at the top.
        strict_dataflow: If `True`, the transformation ensures a more
            stricter version of the data flow.
        apply_fusion_callback: A user supplied function, same signature as `can_be_fused()`,
            to check if a fusion should be performed.

    Note:
        If `strict_dataflow` mode is enabled then the transformation will not remove
        _direct_ data flow dependency from the graph. Furthermore, the transformation
        will not remove size 1 dimensions of intermediate it creates.
        This is a compatibility mode, that will limit the applicability of the
        transformation, but might help transformations that do not fully analyse
        the graph.
    """

    only_toplevel_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are in the top level.",
    )
    only_inner_maps = properties.Property(
        dtype=bool,
        default=False,
        desc="Only perform fusing if the Maps are inner Maps, i.e. does not have top level scope.",
    )
    strict_dataflow = properties.Property(
        dtype=bool,
        default=False,
        desc="If `True` then the transformation will ensure a more stricter data flow.",
    )

    # Callable that can be specified by the user, if it is specified, it should be
    #  a callable with the same signature as `can_be_fused()`. If the function returns
    #  `False` then the fusion will be rejected.
    _apply_fusion_callback: Optional[FusionCallback]

    def __init__(
        self,
        only_inner_maps: Optional[bool] = None,
        only_toplevel_maps: Optional[bool] = None,
        strict_dataflow: Optional[bool] = None,
        apply_fusion_callback: Optional[FusionCallback] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._shared_data = {}  # type: ignore[var-annotated]
        self._apply_fusion_callback = None
        if only_toplevel_maps is not None:
            self.only_toplevel_maps = bool(only_toplevel_maps)
        if only_inner_maps is not None:
            self.only_inner_maps = bool(only_inner_maps)
        if strict_dataflow is not None:
            self.strict_dataflow = bool(strict_dataflow)
        if apply_fusion_callback is not None:
            self._apply_fusion_callback = apply_fusion_callback

    @classmethod
    def expressions(cls) -> bool:
        raise RuntimeError("The `MapFusionHelper` is not a transformation on its own.")

    def can_be_fused(
        self,
        map_entry_1: nodes.MapEntry,
        map_entry_2: nodes.MapEntry,
        graph: Union[dace.SDFGState, dace.SDFG],
        sdfg: dace.SDFG,
        permissive: bool = False,
    ) -> bool:
        """Performs basic checks if the maps can be fused.

        This function only checks constrains that are common between serial and
        parallel map fusion process, which includes:
        - The registered callback, if specified.
        - The scope of the maps.
        - The scheduling of the maps.
        - The map parameters.

        Args:
            map_entry_1: The entry of the first (in serial case the top) map.
            map_exit_2: The entry of the second (in serial case the bottom) map.
            graph: The SDFGState in which the maps are located.
            sdfg: The SDFG itself.
            permissive: Currently unused.
        """
        # Consult the callback if defined.
        if self._apply_fusion_callback is not None:
            if not self._apply_fusion_callback(
                self, map_entry_1, map_entry_2, graph, sdfg, permissive
            ):
                return False

        if self.only_inner_maps and self.only_toplevel_maps:
            raise ValueError("You specified both `only_inner_maps` and `only_toplevel_maps`.")

        # Ensure that both have the same schedule
        if map_entry_1.map.schedule != map_entry_2.map.schedule:
            return False

        # Fusing is only possible if the two entries are in the same scope.
        scope = graph.scope_dict()
        if scope[map_entry_1] != scope[map_entry_2]:
            return False
        elif self.only_inner_maps:
            if scope[map_entry_1] is None:
                return False
        elif self.only_toplevel_maps:
            if scope[map_entry_1] is not None:
                return False

        # We will now check if there exists a remapping of the map parameter
        if (
            self.find_parameter_remapping(first_map=map_entry_1.map, second_map=map_entry_2.map)
            is None
        ):
            return False

        return True

    def relocate_nodes(
        self,
        from_node: Union[nodes.MapExit, nodes.MapEntry],
        to_node: Union[nodes.MapExit, nodes.MapEntry],
        state: SDFGState,
        sdfg: SDFG,
    ) -> None:
        """Move the connectors and edges from `from_node` to `to_nodes` node.

        This function will only rewire the edges, it does not remove the nodes
        themselves. Furthermore, this function should be called twice per Map,
        once for the entry and then for the exit.
        While it does not remove the node themselves if guarantees that the
        `from_node` has degree zero.
        The function assumes that the parameter renaming was already done.

        Args:
            from_node: Node from which the edges should be removed.
            to_node: Node to which the edges should reconnect.
            state: The state in which the operation happens.
            sdfg: The SDFG that is modified.
        """

        # Now we relocate empty Memlets, from the `from_node` to the `to_node`
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.out_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_src=to_node)
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.in_edges(from_node))):
            helpers.redirect_edge(state, empty_edge, new_dst=to_node)

        # We now ensure that there is only one empty Memlet from the `to_node` to any other node.
        #  Although it is allowed, we try to prevent it.
        empty_targets: Set[nodes.Node] = set()
        for empty_edge in list(filter(lambda e: e.data.is_empty(), state.all_edges(to_node))):
            if empty_edge.dst in empty_targets:
                state.remove_edge(empty_edge)
            empty_targets.add(empty_edge.dst)

        # We now determine which edges we have to migrate, for this we are looking at
        #  the incoming edges, because this allows us also to detect dynamic map ranges.
        #  TODO(phimuell): If there is already a connection to the node, reuse this.
        for edge_to_move in list(state.in_edges(from_node)):
            assert isinstance(edge_to_move.dst_conn, str)

            if not edge_to_move.dst_conn.startswith("IN_"):
                # Dynamic Map Range
                #  The connector name simply defines a variable name that is used,
                #  inside the Map scope to define a variable. We handle it directly.
                dmr_symbol = edge_to_move.dst_conn

                # TODO(phimuell): Check if the symbol is really unused in the target scope.
                if dmr_symbol in to_node.in_connectors:
                    raise NotImplementedError(
                        f"Tried to move the dynamic map range '{dmr_symbol}' from {from_node}'"
                        f" to '{to_node}', but the symbol is already known there, but the"
                        " renaming is not implemented."
                    )
                if not to_node.add_in_connector(dmr_symbol, force=False):
                    raise RuntimeError(  # Might fail because of out connectors.
                        f"Failed to add the dynamic map range symbol '{dmr_symbol}' to '{to_node}'."
                    )
                helpers.redirect_edge(state=state, edge=edge_to_move, new_dst=to_node)
                from_node.remove_in_connector(dmr_symbol)

            else:
                # We have a Passthrough connection, i.e. there exists a matching `OUT_`.
                old_conn = edge_to_move.dst_conn[3:]  # The connection name without prefix
                new_conn = to_node.next_connector(old_conn)

                to_node.add_in_connector("IN_" + new_conn)
                for e in list(state.in_edges_by_connector(from_node, "IN_" + old_conn)):
                    helpers.redirect_edge(state, e, new_dst=to_node, new_dst_conn="IN_" + new_conn)
                to_node.add_out_connector("OUT_" + new_conn)
                for e in list(state.out_edges_by_connector(from_node, "OUT_" + old_conn)):
                    helpers.redirect_edge(state, e, new_src=to_node, new_src_conn="OUT_" + new_conn)
                from_node.remove_in_connector("IN_" + old_conn)
                from_node.remove_out_connector("OUT_" + old_conn)

        # Check if we succeeded.
        if state.out_degree(from_node) != 0:
            raise validation.InvalidSDFGError(
                f"Failed to relocate the outgoing edges from `{from_node}`, there are still `{state.out_edges(from_node)}`",
                sdfg,
                sdfg.node_id(state),
            )
        if state.in_degree(from_node) != 0:
            raise validation.InvalidSDFGError(
                f"Failed to relocate the incoming edges from `{from_node}`, there are still `{state.in_edges(from_node)}`",
                sdfg,
                sdfg.node_id(state),
            )
        assert len(from_node.in_connectors) == 0
        assert len(from_node.out_connectors) == 0

    def find_parameter_remapping(
        self, first_map: nodes.Map, second_map: nodes.Map
    ) -> Union[Dict[str, str], None]:
        """Computes the parameter remapping for the parameters of the _second_ map.

        The returned `dict` maps the parameters of the second map (keys) to parameter
        names of the first map (values). Because of how the replace function works
        the `dict` describes how to replace the parameters of the second map
        with parameters of the first map.
        Parameters that already have the correct name and compatible range, are not
        included in the return value, thus the keys and values are always different.
        If no renaming at all is _needed_, i.e. all parameter have the same name and
        range, then the function returns an empty `dict`.
        If no remapping exists, then the function will return `None`.

        Args:
            first_map:  The first map (these parameters will be replaced).
            second_map: The second map, these parameters acts as source.
        """

        # The parameter names
        first_params: List[str] = first_map.params
        second_params: List[str] = second_map.params

        if len(first_params) != len(second_params):
            return None

        # The ranges, however, we apply some post processing to them.
        simp = lambda e: symbolic.simplify_ext(symbolic.simplify(e))  # noqa: E731
        first_rngs: Dict[str, Tuple[Any, Any, Any]] = {
            param: tuple(simp(r) for r in rng) for param, rng in zip(first_params, first_map.range)
        }
        second_rngs: Dict[str, Tuple[Any, Any, Any]] = {
            param: tuple(simp(r) for r in rng)
            for param, rng in zip(second_params, second_map.range)
        }

        # Parameters of the second map that have not yet been matched to a parameter
        #  of the first map and vice versa.
        unmapped_second_params: Set[str] = set(second_params)
        unused_first_params: Set[str] = set(first_params)

        # This is the result (`second_param -> first_param`), note that if no renaming
        #  is needed then the parameter is not present in the mapping.
        final_mapping: Dict[str, str] = {}

        # First we identify the parameters that already have the correct name.
        for param in set(first_params).intersection(second_params):
            first_rng = first_rngs[param]
            second_rng = second_rngs[param]

            if first_rng == second_rng:
                # They have the same name and the same range, this is already a match.
                #  Because the names are already the same, we do not have to enter them
                #  in the `final_mapping`
                unmapped_second_params.discard(param)
                unused_first_params.discard(param)

        # Check if no remapping is needed.
        if len(unmapped_second_params) == 0:
            return {}

        # Now we go through all the parameters that we have not mapped yet.
        #  All of them will result in a remapping.
        for unmapped_second_param in unmapped_second_params:
            second_rng = second_rngs[unmapped_second_param]
            assert unmapped_second_param not in final_mapping

            # Now look in all not yet used parameters of the first map which to use.
            for candidate_param in unused_first_params:
                candidate_rng = first_rngs[candidate_param]
                if candidate_rng == second_rng:
                    final_mapping[unmapped_second_param] = candidate_param
                    unused_first_params.discard(candidate_param)
                    break
            else:
                # We did not find a candidate, so the remapping does not exist
                return None

        assert len(unused_first_params) == 0
        assert len(final_mapping) == len(unmapped_second_params)
        return final_mapping

    def rename_map_parameters(
        self,
        first_map: nodes.Map,
        second_map: nodes.Map,
        second_map_entry: nodes.MapEntry,
        state: SDFGState,
    ) -> None:
        """Replaces the map parameters of the second map with names from the first.

        The replacement is done in a safe way, thus `{'i': 'j', 'j': 'i'}` is
        handled correct. The function assumes that a proper replacement exists.
        The replacement is computed by calling `self.find_parameter_remapping()`.

        Args:
            first_map:  The first map (these are the final parameter).
            second_map: The second map, this map will be replaced.
            second_map_entry: The entry node of the second map.
            state: The SDFGState on which we operate.
        """
        # Compute the replacement dict.
        repl_dict: Dict[str, str] = self.find_parameter_remapping(  # type: ignore[assignment]
            first_map=first_map, second_map=second_map
        )

        if repl_dict is None:
            raise RuntimeError("The replacement does not exist")
        if len(repl_dict) == 0:
            return

        second_map_scope = state.scope_subgraph(entry_node=second_map_entry)
        # Why is this thing is symbolic and not in replace?
        symbolic.safe_replace(
            mapping=repl_dict,
            replace_callback=second_map_scope.replace_dict,
        )

        # For some odd reason the replace function does not modify the range and
        #  parameter of the map, so we will do it the hard way.
        second_map.params = copy.deepcopy(first_map.params)
        second_map.range = copy.deepcopy(first_map.range)

    def is_shared_data(
        self,
        data: nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """Tests if `data` is shared data, i.e. it can not be removed from the SDFG.

        Depending on the situation, the function will not perform a scan of the whole SDFG:
        1) If `data` is non transient then the function will return `True`, as non transient data
            must be reconstructed always.
        2) If the AccessNode `data` has more than one outgoing edge or more than one incoming edge
            it is classified as shared.
        3) If `FindSingleUseData` is in the pipeline it will be used and no scan will be performed.
        4) The function will perform a scan.

        :param data: The transient that should be checked.
        :param state: The state in which the fusion is performed.
        :param sdfg: The SDFG in which we want to perform the fusing.

        """
        # If `data` is non transient then return `True` as the intermediate can not be removed.
        if not data.desc(sdfg).transient:
            return True

        # This means the data is consumed by multiple Maps, through the same AccessNode, in this state
        #  Note currently multiple incoming edges are not handled, but in the spirit of this function
        #  we consider such AccessNodes as shared, because we can not remove the intermediate.
        if state.out_degree(data) > 1:
            return True
        if state.in_degree(data) > 1:
            return True

        # We have to perform the full scan of the SDFG.
        return self._scan_sdfg_if_data_is_shared(data=data, state=state, sdfg=sdfg)

    def _scan_sdfg_if_data_is_shared(
        self,
        data: nodes.AccessNode,
        state: dace.SDFGState,
        sdfg: dace.SDFG,
    ) -> bool:
        """Scans `sdfg` to determine if `data` is shared.

        Essentially, this function determines if the intermediate AccessNode `data`
        can be removed or if it has to be restored as output of the Map.
        A data descriptor is classified as shared if any of the following is true:
        - `data` is non transient data.
        - `data` has at most one incoming and/or outgoing edge.
        - There are other AccessNodes beside `data` that refer to the same data.
        - The data is accessed on an interstate edge.

        This function should not be called directly. Instead it is called indirectly
        by `is_shared_data()` if there is no short cut.

        :param data: The AccessNode that should checked if it is shared.
        :param sdfg: The SDFG for which the set of shared data should be computed.
        """
        if not data.desc(sdfg).transient:
            return True

        # See description in `is_shared_data()` for more.
        if state.out_degree(data) > 1:
            return True
        if state.in_degree(data) > 1:
            return True

        data_name: str = data.data
        for state in sdfg.states():
            for dnode in state.data_nodes():
                if dnode is data:
                    # We have found the `data` AccessNode, which we must ignore.
                    continue
                if dnode.data == data_name:
                    # We found a different AccessNode that refers to the same data
                    #  as `data`. Thus `data` is shared.
                    return True

        # Test if the data is referenced in the interstate edges.
        for edge in sdfg.edges():
            if data_name in edge.data.free_symbols:
                # The data is used in the inter state edges. So it is shared.
                return True

        # Test if they are accessed in a condition of a loop or conditional block.
        for cfr in sdfg.all_control_flow_regions():
            if data_name in cfr.used_symbols(all_symbols=True, with_contents=False):
                return True

        # The `data` is not used anywhere else, thus `data` is not shared.
        return False

    def _compute_multi_write_data(
        self,
        state: SDFGState,
        sdfg: SDFG,
    ) -> Set[str]:
        """Computes data inside a _single_ state, that is written multiple times.

        Essentially this function computes the set of data that does not follow
        the single static assignment idiom. The function also resolves views.
        If an access node, refers to a view, not only the view itself, but also
        the data it refers to is added to the set.

        Args:
            state: The state that should be examined.
            sdfg: The SDFG object.

        Note:
            This information is used by the partition function (in case strict data
            flow mode is enabled), in strict data flow mode only. The current
            implementation is rather simple as it only checks if a data is written
            to multiple times in the same state.
        """
        data_written_to: Set[str] = set()
        multi_write_data: Set[str] = set()

        for access_node in state.data_nodes():
            if state.in_degree(access_node) == 0:
                continue
            if access_node.data in data_written_to:
                multi_write_data.add(access_node.data)
            elif self.is_view(access_node, sdfg):
                # This is an over approximation.
                multi_write_data.update(
                    [access_node.data, self.track_view(access_node, state, sdfg).data]
                )
            data_written_to.add(access_node.data)
        return multi_write_data

    def is_node_reachable_from(
        self,
        graph: dace.SDFGState,
        begin: nodes.Node,
        end: nodes.Node,
    ) -> bool:
        """Test if the node `end` can be reached from `begin`.

        Essentially the function starts a DFS at `begin`. If an edge is found that lead
        to `end` the function returns `True`. If the node is never found `False` is
        returned.

        Args:
            graph: The graph to operate on.
            begin: The start of the DFS.
            end: The node that should be located.
        """

        def next_nodes(node: nodes.Node) -> Iterable[nodes.Node]:
            return (edge.dst for edge in graph.out_edges(node))

        to_visit: List[nodes.Node] = [begin]
        seen: Set[nodes.Node] = set()

        while len(to_visit) > 0:
            node: nodes.Node = to_visit.pop()
            if node == end:
                return True
            elif node not in seen:
                to_visit.extend(next_nodes(node))
            seen.add(node)

        # We never found `end`
        return False

    def get_access_set(
        self,
        scope_node: Union[nodes.MapEntry, nodes.MapExit],
        state: SDFGState,
    ) -> Set[nodes.AccessNode]:
        """Computes the access set of a "scope node".

        If `scope_node` is a `MapEntry` it will operate on the set of incoming edges
        and if it is an `MapExit` on the set of outgoing edges. The function will
        then determine all access nodes that have a connection through these edges
        to the scope nodes (edges that does not lead to access nodes are ignored).
        The function returns a set that contains all access nodes that were found.
        It is important that this set will also contain views.

        Args:
            scope_node: The scope node that should be evaluated.
            state: The state in which we operate.
        """
        if isinstance(scope_node, nodes.MapEntry):
            get_edges = lambda node: state.in_edges(node)  # noqa: E731
            other_node = lambda e: e.src  # noqa: E731
        else:
            get_edges = lambda node: state.out_edges(node)  # noqa: E731
            other_node = lambda e: e.dst  # noqa: E731
        access_set: Set[nodes.AccessNode] = {
            node
            for node in map(other_node, get_edges(scope_node))
            if isinstance(node, nodes.AccessNode)
        }

        return access_set

    def find_subsets(
        self,
        node: nodes.AccessNode,
        scope_node: Union[nodes.MapExit, nodes.MapEntry],
        state: SDFGState,
        sdfg: SDFG,
        repl_dict: Optional[Dict[str, str]],
    ) -> List[subsets.Subset]:
        """Finds all subsets that access `node` within `scope_node`.

        The function will not start a search for all consumer/producers.
        Instead it will locate the edges which is immediately inside the
        map scope.

        Args:
            node: The access node that should be examined.
            scope_node: We are only interested in data that flows through this node.
            state: The state in which we operate.
            sdfg: The SDFG object.
        """

        # Is the node used for reading or for writing.
        #  This influences how we have to proceed.
        if isinstance(scope_node, nodes.MapEntry):
            outer_edges_to_inspect = [e for e in state.in_edges(scope_node) if e.src == node]
            get_subset = lambda e: e.data.src_subset  # noqa: E731
            get_inner_edges = lambda e: state.out_edges_by_connector(
                scope_node, "OUT_" + e.dst_conn[3:]
            )
        else:
            outer_edges_to_inspect = [e for e in state.out_edges(scope_node) if e.dst == node]
            get_subset = lambda e: e.data.dst_subset  # noqa: E731
            get_inner_edges = lambda e: state.in_edges_by_connector(
                scope_node, "IN_" + e.src_conn[4:]
            )

        found_subsets: List[subsets.Subset] = []
        for edge in outer_edges_to_inspect:
            found_subsets.extend(get_subset(e) for e in get_inner_edges(edge))
        assert len(found_subsets) > 0, "Could not find any subsets."
        assert not any(subset is None for subset in found_subsets)

        found_subsets = copy.deepcopy(found_subsets)
        if repl_dict:
            for subset in found_subsets:
                # Replace happens in place
                symbolic.safe_replace(repl_dict, subset.replace)

        return found_subsets

    def is_view(
        self,
        node: nodes.AccessNode,
        sdfg: SDFG,
    ) -> bool:
        """Tests if `node` points to a view or not."""
        node_desc: data.Data = node.desc(sdfg)
        return isinstance(node_desc, data.View)

    def track_view(
        self,
        view: nodes.AccessNode,
        state: SDFGState,
        sdfg: SDFG,
    ) -> nodes.AccessNode:
        """Find the original data of a View.

        Given the View `view`, the function will trace the view back to the original
        access node. For convenience, if `view` is not a `View` the argument will be
        returned.

        Args:
            view: The view that should be traced.
            state: The state in which we operate.
            sdfg: The SDFG on which we operate.
        """

        # Test if it is a view at all, if not return the passed node as source.
        if not self.is_view(view, sdfg):
            return view

        # First determine if the view is used for reading or writing.
        curr_edge = dace.sdfg.utils.get_view_edge(state, view)
        if curr_edge is None:
            raise RuntimeError(f"Failed to determine the direction of the view '{view}'.")
        if curr_edge.dst_conn == "views":
            # The view is used for reading.
            next_node = lambda curr_edge: curr_edge.src  # noqa: E731
        elif curr_edge.src_conn == "views":
            # The view is used for writing.
            next_node = lambda curr_edge: curr_edge.dst  # noqa: E731
        else:
            raise RuntimeError(
                f"Failed to determine the direction of the view '{view}' | {curr_edge}."
            )

        # Now trace the view back.
        org_view = view
        view = next_node(curr_edge)
        while self.is_view(view, sdfg):
            curr_edge = dace.sdfg.utils.get_view_edge(state, view)
            if curr_edge is None:
                raise RuntimeError(f"View tracing of '{org_view}' failed at note '{view}'.")
            view = next_node(curr_edge)
        return view
