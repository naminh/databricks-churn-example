from setuptools import find_packages, setup
from churn import __version__

setup(
    name="churn",
    packages=find_packages(exclude=["tests", "tests.*"]),
    setup_requires=["wheel"],
    version=__version__,
    description="churn",
    author="Minh Nguyen",
    python_requires=">=3.8",
)