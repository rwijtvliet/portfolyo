from distutils.core import setup
import versioneer

setup(
    name="portfolyo",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Ruud Wijtvliet",
    packages=["portfolio"],
    description="Analysing and manipulating timeseries related to power and gas offtake portfolios.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    python_requires=">=3.9",
    install_requires=[line.strip() for line in open("requirements.txt", "r")],
)
