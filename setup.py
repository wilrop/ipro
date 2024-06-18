from distutils.core import setup

setup(
    name="IPRO",
    version="0.1",
    description="Iterated Pareto Referent Optimisation",
    packages=["ipro"],
    author="Willem Röpke",
    install_requires=[],
    extras_require={
        "dev": [],
    },
    python_requires=">=3.10",
)
