from distutils.core import setup

setup(
    name="portfolyo",
    version="0.0.1",
    author="Ruud Wijtvliet",
    packages=["portfolio"],
    description="Analysing and manipulating timeseries related to power and gas offtake portfolios.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    python_requires=">=3.8",
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
)
