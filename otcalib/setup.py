"""
This script is used to install the package and all its dependencies. Run

    python -m pip install .

to install the package.
"""

from setuptools import setup

setup(
    name="otcalib",
    version="0.0",  # Also change in module
    packages=[
        "otcalib",
        "otcalib.evaluate",
        "otcalib.torch",
        "otcalib.utils",
        # "otcalib.data",
    ],
    install_requires=[
            "-e ./otcalib",  # Link the otcalib submodule as an editable dependency
        ]
    include_package_data=True,
    test_suite="otcalib.tests",
    scripts=[],
    description="Optimal Transport framework for ATLAS",
    url="https://gitlab.cern.ch/aml/optimal-transport/ot-framework",
)
