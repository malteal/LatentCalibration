import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tools",
    version="0.0.1",
    description="Some common tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=["tools"],
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "pandas",
        "PIL",
        "setuptools",
        "tqdm",
    ],
)
