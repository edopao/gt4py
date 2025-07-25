# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import abc
import os
import textwrap
import time
from typing import TYPE_CHECKING, Any, Final

from gt4py.cartesian import backend as gt_backend, config as gt_config, utils as gt_utils
from gt4py.cartesian.backend import Backend
from gt4py.cartesian.backend.module_generator import BaseModuleGenerator, ModuleData
from gt4py.cartesian.gtc import gtir, utils as gtc_utils
from gt4py.cartesian.gtc.passes.oir_pipeline import OirPipeline
from gt4py.eve import codegen


if TYPE_CHECKING:
    from gt4py.cartesian.stencil_builder import StencilBuilder
    from gt4py.cartesian.stencil_object import StencilObject


def pybuffer_to_sid(
    *,
    name: str,
    ctype: str,
    domain_dim_flags: tuple[bool, bool, bool],
    data_ndim: int,
    stride_kind_index: int,
    backend: Backend,
):
    domain_ndim = domain_dim_flags.count(True)
    sid_ndim = domain_ndim + data_ndim

    as_sid = "as_cuda_sid" if backend.storage_info["device"] == "gpu" else "as_sid"

    sid_def = """gt::{as_sid}<{ctype}, {sid_ndim},
        gt::integral_constant<int, {unique_index}>>({name})""".format(
        name=name, ctype=ctype, unique_index=stride_kind_index, sid_ndim=sid_ndim, as_sid=as_sid
    )
    sid_def = "gt::sid::shift_sid_origin({sid_def}, {name}_origin)".format(
        sid_def=sid_def, name=name
    )
    if domain_ndim != 3:
        gt_dims = [
            f"gt::stencil::dim::{dim}"
            for dim in gtc_utils.dimension_flags_to_names(domain_dim_flags)
        ]
        if data_ndim:
            gt_dims += [f"gt::integral_constant<int, {3 + dim}>" for dim in range(data_ndim)]
        sid_def = "gt::sid::rename_numbered_dimensions<{gt_dims}>({sid_def})".format(
            gt_dims=", ".join(gt_dims), sid_def=sid_def
        )

    return sid_def


def bindings_main_template():
    return codegen.MakoTemplate(
        """
        #include <chrono>
        #include <pybind11/pybind11.h>
        #include <pybind11/stl.h>
        #include <gridtools/storage/adapter/python_sid_adapter.hpp>
        #include <gridtools/stencil/cartesian.hpp>
        #include <gridtools/stencil/global_parameter.hpp>
        #include <gridtools/sid/sid_shift_origin.hpp>
        #include <gridtools/sid/rename_dimensions.hpp>
        #include "computation.hpp"
        namespace gt = gridtools;
        namespace py = ::pybind11;
        PYBIND11_MODULE(${module_name}, m) {
            m.def("run_computation", [](
            ${','.join(["std::array<gt::uint_t, 3> domain", *entry_params, 'py::object exec_info'])}
            ){
                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_start_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count())/1e9;
                }

                ${name}(domain)(${','.join(sid_params)});

                if (!exec_info.is(py::none()))
                {
                    auto exec_info_dict = exec_info.cast<py::dict>();
                    exec_info_dict["run_cpp_end_time"] = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch()).count()/1e9);
                }

            }, "Runs the given computation");}
        """
    )


class PyExtModuleGenerator(BaseModuleGenerator):
    """Module Generator for use with backends that generate c++ python extensions."""

    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

    def __call__(self, args_data: ModuleData) -> str:
        return super().__call__(args_data)

    def _is_not_empty(self) -> bool:
        node = self.builder.gtir_pipeline.full()
        return bool(node.walk_values().if_isinstance(gtir.ParAssignStmt).to_list())

    def generate_imports(self) -> str:
        source = [
            *super().generate_imports().splitlines(),
            "from gt4py.cartesian import utils as gt_utils",
        ]
        if self._is_not_empty():
            backend_data = self.builder.backend_data

            file_path = 'f"{{pathlib.Path(__file__).parent.resolve()}}/{}"'.format(
                os.path.basename(backend_data["pyext_file_path"])
            )
            source.append(
                textwrap.dedent(
                    f"""
                pyext_module = gt_utils.make_module_from_file(
                    "{backend_data["pyext_module_name"]}", {file_path}, public_import=True
                )
                """
                )
            )
        return "\n".join(source)

    def _has_effect(self) -> bool:
        return self._is_not_empty()

    def generate_implementation(self) -> str:
        ir = self.builder.gtir
        sources = codegen.TextBlock(indent_size=BaseModuleGenerator.TEMPLATE_INDENT_SIZE)

        args: list[str] = []
        for decl in ir.params:
            args.append(decl.name)
            if isinstance(decl, gtir.FieldDecl):
                args.append("list(_origin_['{}'])".format(decl.name))

        # only generate implementation if any multi_stages are present. e.g. if no statement in the
        # stencil has any effect on the API fields, this may not be the case since they could be
        # pruned.
        if self._has_effect():
            source = textwrap.dedent(
                f"""
                # Load or generate a GTComputation object for the current domain size
                pyext_module.run_computation({",".join(["list(_domain_)", *args, "exec_info"])})
                """
            )
            sources.extend(source.splitlines())
        else:
            sources.extend("\n")

        return sources.text


class BackendCodegen:
    TEMPLATE_FILES: dict[str, str]

    @abc.abstractmethod
    def __init__(self, class_name: str, module_name: str, backend: Backend) -> None:
        pass

    @abc.abstractmethod
    def __call__(self) -> dict[str, dict[str, str]]:
        """Return a dict with the keys 'computation' and 'bindings' to dicts of filenames to source."""
        pass


GTBackendOptions = dict[str, dict[str, Any]]


class BaseGTBackend(gt_backend.BasePyExtBackend):
    GT_BACKEND_OPTS: Final[GTBackendOptions] = {
        "add_profile_info": {"versioning": True, "type": bool},
        "clean": {"versioning": False, "type": bool},
        "debug_mode": {"versioning": True, "type": bool},
        "opt_level": {"versioning": True, "type": str},
        "extra_opt_flags": {"versioning": True, "type": str},
        "verbose": {"versioning": False, "type": bool},
        "oir_pipeline": {"versioning": True, "type": OirPipeline},
    }

    GT_BACKEND_T: str

    MODULE_GENERATOR_CLASS = PyExtModuleGenerator

    PYEXT_GENERATOR_CLASS: type[BackendCodegen]

    @abc.abstractmethod
    def generate(self) -> type[StencilObject]:
        pass

    @abc.abstractmethod
    def generate_extension(self) -> None:
        """
        Generate and build a python extension for the stencil computation.
        """
        pass

    def make_extension(self, *, uses_cuda: bool = False) -> None:
        build_info = self.builder.options.build_info
        if build_info is not None:
            start_time = time.perf_counter()

        # Generate source
        gt_pyext_files: dict[str, Any]
        gt_pyext_sources: dict[str, Any]
        if self.builder.options._impl_opts.get("disable-code-generation", False):
            # Pass NOTHING to the self.builder means try to reuse the source code files
            gt_pyext_files = {}
            gt_pyext_sources = {
                key: gt_utils.NOTHING for key in self.PYEXT_GENERATOR_CLASS.TEMPLATE_FILES.keys()
            }
        else:
            gt_pyext_files = self._make_extension_sources()
            gt_pyext_sources = {
                **gt_pyext_files["computation"],
                **gt_pyext_files["bindings"],
            }

        if build_info is not None:
            next_time = time.perf_counter()
            build_info["codegen_time"] = next_time - start_time
            start_time = next_time

        # Build extension module
        pyext_opts = dict(
            verbose=self.builder.options.backend_opts.get("verbose", False),
            clean=self.builder.options.backend_opts.get("clean", False),
            **gt_backend.pyext_builder.get_gt_pyext_build_opts(
                debug_mode=self.builder.options.backend_opts.get("debug_mode", False),
                opt_level=self.builder.options.backend_opts.get(
                    "opt_level", gt_config.GT4PY_COMPILE_OPT_LEVEL
                ),
                extra_opt_flags=self.builder.options.backend_opts.get(
                    "extra_opt_flags", gt_config.GT4PY_EXTRA_COMPILE_OPT_FLAGS
                ),
                add_profile_info=self.builder.options.backend_opts.get("add_profile_info", False),
                uses_cuda=uses_cuda,
            ),
        )

        self.build_extension_module(gt_pyext_sources, pyext_opts, uses_cuda=uses_cuda)

        if build_info is not None:
            build_info["build_time"] = time.perf_counter() - start_time

    def _make_extension_sources(self) -> dict[str, dict[str, str]]:
        """Generate the source for the stencil independently from use case."""
        if "computation_src" in self.builder.backend_data:
            return self.builder.backend_data["computation_src"]

        class_name = self.pyext_class_name if self.builder.stencil_id else self.builder.options.name
        module_name = (
            self.pyext_module_name
            if self.builder.stencil_id
            else f"{self.builder.options.name}_pyext"
        )
        gt_pyext_generator = self.PYEXT_GENERATOR_CLASS(class_name, module_name, self)
        gt_pyext_sources = gt_pyext_generator()
        final_ext = ".cu" if self.languages and self.languages["computation"] == "cuda" else ".cpp"
        comp_src = gt_pyext_sources["computation"]
        for key in [k for k in comp_src.keys() if k.endswith(".src")]:
            comp_src[key.replace(".src", final_ext)] = comp_src.pop(key)
        self.builder.backend_data["computation_src"] = gt_pyext_sources
        return gt_pyext_sources


class CUDAPyExtModuleGenerator(PyExtModuleGenerator):
    def __init__(self, builder: StencilBuilder) -> None:
        super().__init__(builder)

    def generate_implementation(self) -> str:
        source = super().generate_implementation()
        if self.builder.options.backend_opts.get("device_sync", True):
            source += textwrap.dedent(
                """
                    cupy.cuda.Device(0).synchronize()
                """
            )
        return source

    def generate_imports(self) -> str:
        source = (
            textwrap.dedent(
                """
                import cupy
                """
            )
            + super().generate_imports()
        )
        return source
