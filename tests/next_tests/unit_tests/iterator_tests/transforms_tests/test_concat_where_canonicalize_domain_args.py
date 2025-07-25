# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import common
import pytest
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import (
    concat_where,
    inline_lambdas,
)
from gt4py.next.type_system import type_specifications as ts

int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
field_type = ts.FieldType(dims=[IDim], dtype=int_type)


def cases():
    return [
        # testee, expected
        (
            im.concat_where(im.and_("cond1", "cond2"), "a", "b"),
            im.concat_where("cond1", im.concat_where("cond2", "a", "b"), "b"),
        ),
        (
            im.concat_where(im.or_("cond1", "cond2"), "a", "b"),
            im.concat_where("cond1", "a", im.concat_where("cond2", "a", "b")),
        ),
        (
            im.concat_where(im.domain(common.GridType.CARTESIAN, {IDim: (0, 1)}), "a", "b"),
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (itir.InfinityLiteral.NEGATIVE, 0)}),
                "b",
                im.concat_where(
                    im.domain(
                        common.GridType.CARTESIAN, {IDim: (1, itir.InfinityLiteral.POSITIVE)}
                    ),
                    "b",
                    "a",
                ),
            ),
        ),
        (  # not transformed
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (0, itir.InfinityLiteral.POSITIVE)}),
                "a",
                "b",
            ),
            im.concat_where(
                im.domain(common.GridType.CARTESIAN, {IDim: (0, itir.InfinityLiteral.POSITIVE)}),
                "a",
                "b",
            ),
        ),
    ]


@pytest.mark.parametrize("testee, expected", cases())
def test_nested_concat_where(testee, expected):
    actual = concat_where.canonicalize_domain_argument(testee)
    actual = inline_lambdas.InlineLambdas.apply(actual, opcount_preserving=True)

    assert actual == expected
