import os
import sys

from setuptools import setup


CUSTOM_KERNEL_FLAG = "--with-custom-kernel"
TRUTHY = {"1", "true", "yes", "on"}
DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET = "15.0"


def _with_custom_kernel() -> bool:
    if CUSTOM_KERNEL_FLAG in sys.argv:
        sys.argv.remove(CUSTOM_KERNEL_FLAG)
        return True
    return os.environ.get("OMLX_WITH_CUSTOM_KERNEL", "").strip().lower() in TRUTHY


# MLX vendors nanobind via FetchContent (a pinned GIT_TAG in its CMakeLists),
# and our custom-kernel extensions share MLX's `mlx` NB_DOMAIN. If the nanobind
# used to build the extension differs (by ABI) from the one MLX was built with,
# the build SUCCEEDS but every `mlx.core.array` is rejected at the type caster
# ("incompatible function arguments"), so the native kernel silently disables
# itself at import (fast.has_native() -> False). Fail loudly at build time with
# the exact version to install instead of shipping a dead kernel.
#
# Map of MLX minor version -> the nanobind tag it pins:
#   0.32.x  (stock upstream)      -> 2.13.0
#   0.31.x  (PrismML 1-bit fork)  -> 2.12.0
# Extend as new MLX builds are targeted; unknown MLX versions pass through.
_MLX_NANOBIND_ABI = {
    "0.32": "2.13.0",
    "0.31": "2.12.0",
}


def _verify_nanobind_abi() -> None:
    try:
        from importlib import metadata as _md

        mlx_ver = _md.version("mlx")
        nb_ver = _md.version("nanobind")
    except Exception:
        # Can't introspect (e.g. build isolation) — defer to build-system pins.
        return

    want = _MLX_NANOBIND_ABI.get(".".join(mlx_ver.split(".")[:2]))
    if want and nb_ver != want:
        raise SystemExit(
            "omlx custom-kernel build: nanobind ABI mismatch.\n"
            f"  installed mlx      = {mlx_ver} (built against nanobind {want})\n"
            f"  installed nanobind = {nb_ver}\n"
            "Building against this nanobind would produce a kernel that rejects "
            "every mlx.core.array at runtime (isolated `mlx` NB_DOMAIN), so "
            "fast.has_native() stays False. Fix:\n"
            f"    pip install 'nanobind=={want}'\n"
            "then rebuild with OMLX_WITH_CUSTOM_KERNEL=1."
        )


def _custom_kernel_build_kwargs() -> dict:
    if not _with_custom_kernel():
        return {}

    _verify_nanobind_abi()

    target = (
        os.environ.get("OMLX_CUSTOM_KERNEL_DEPLOYMENT_TARGET")
        or os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        or DEFAULT_CUSTOM_KERNEL_DEPLOYMENT_TARGET
    )
    os.environ.setdefault("MACOSX_DEPLOYMENT_TARGET", target)
    cmake_args = os.environ.get("CMAKE_ARGS", "").strip()
    if "CMAKE_OSX_DEPLOYMENT_TARGET" not in cmake_args:
        target_arg = f"-DCMAKE_OSX_DEPLOYMENT_TARGET={target}"
        os.environ["CMAKE_ARGS"] = (
            f"{cmake_args} {target_arg}".strip() if cmake_args else target_arg
        )

    from mlx import extension

    return {
        "ext_modules": [
            extension.CMakeExtension(
                "omlx.custom_kernels.bonsai._ext",
                sourcedir="omlx/custom_kernels/bonsai/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.glm_moe_dsa._ext",
                sourcedir="omlx/custom_kernels/glm_moe_dsa/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.minimax_m3._ext",
                sourcedir="omlx/custom_kernels/minimax_m3/csrc",
            ),
            extension.CMakeExtension(
                "omlx.custom_kernels.qwen35_prefill._ext",
                sourcedir="omlx/custom_kernels/qwen35_prefill/csrc",
            ),
        ],
        "cmdclass": {"build_ext": extension.CMakeBuild},
    }


if __name__ == "__main__":
    setup(**_custom_kernel_build_kwargs())
