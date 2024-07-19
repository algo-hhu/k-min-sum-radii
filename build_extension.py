import os
import subprocess
import warnings
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


def check_openmp_support() -> bool:
    openmp_test_code = """
    #include <omp.h>
    #include <stdio.h>
    int main() {
        int nthreads;
        #pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
        printf("Number of threads = %d\\n", nthreads);
        return 0;
    }
    """

    with open("test_openmp.c", "w") as f:
        f.write(openmp_test_code)

    try:
        # Try to compile the code with OpenMP support
        result = subprocess.run(
            ["gcc", "-fopenmp", "test_openmp.c", "-o", "test_openmp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            return False

        # Run the compiled program
        result = subprocess.run(
            ["./test_openmp"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0:
            return True
        else:
            return False
    finally:
        os.remove("test_openmp.c")
        if os.path.exists("test_openmp"):
            os.remove("test_openmp")


class BuildExt(build_ext):
    """A custom build extension for adding -stdlib arguments for clang++."""

    def build_extensions(self) -> None:
        support = check_openmp_support()

        # '-std=c++11' is added to `extra_compile_args` so the code can compile
        # with clang++. This works across compilers (ignored by MSVC).
        for extension in self.extensions:
            extension.extra_compile_args.append("-std=c++11")
            if support:
                extension.extra_compile_args.append("-fopenmp")
                extension.extra_link_args.append("-lgomp")
            else:
                warnings.warn(
                    "\x1b[31;20m OpenMP is not installed on this system. "
                    "Please install it to have all the benefits from the program.\x1b[0m"
                )

        try:
            build_ext.build_extensions(self)
        except CompileError:
            # Workaround Issue #2.
            # '-stdlib=libc++' is added to `extra_compile_args` and `extra_link_args`
            # so the code can compile on macOS with Anaconda.
            for extension in self.extensions:
                extension.extra_compile_args.append("-stdlib=libc++")
                extension.extra_link_args.append("-stdlib=libc++")
            build_ext.build_extensions(self)


def build(setup_kwargs: Dict[str, Any]) -> None:
    setup_kwargs.update(
        {"ext_modules": [extension], "cmdclass": {"build_ext": BuildExt}}
    )
