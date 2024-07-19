from typing import Any, Dict

from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError

extension = Extension(
    name="kmsr._core",
    sources=[
        "kmsr/_core.cpp",
        "kmsr/cpp/ball.cpp",
        "kmsr/cpp/cluster.cpp",
        "kmsr/cpp/gonzales.cpp",
        "kmsr/cpp/heuristic.cpp",
        "kmsr/cpp/util.cpp",
        "kmsr/cpp/k_MSR.cpp",
        "kmsr/cpp/point.cpp",
        "kmsr/cpp/welzl.cpp",
        "kmsr/cpp/yildirim.cpp",
    ],
    include_dirs=["kmsr"],
)

# Thank you https://github.com/dstein64/kmeans1d!


class BuildExt(build_ext):
    """A custom build extension for adding -stdlib arguments for clang++."""

    def build_extensions(self) -> None:
        # '-std=c++11' is added to `extra_compile_args` so the code can compile
        # with clang++. This works across compilers (ignored by MSVC).
        for extension in self.extensions:
            extension.extra_compile_args.append("-std=c++11")
            extension.extra_compile_args.append("-fopenmp")

        try:
            build_ext.build_extensions(self)
        except CompileError:
            # Workaround Issue #2.
            # '-stdlib=libc++' is added to `extra_compile_args` and `extra_link_args`
            # so the code can compile on macOS with Anaconda.
            for extension in self.extensions:
                extension.extra_compile_args.append("-stdlib=libc++")
                extension.extra_link_args.append("-stdlib=libc++")
                extension.extra_link_args.append("-lomp")
            build_ext.build_extensions(self)


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {"ext_modules": [extension], "cmdclass": {"build_ext": BuildExt}}
    )
