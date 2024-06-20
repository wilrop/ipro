import os
from distutils.core import setup


def _parse_requirements(path: str) -> list[str]:
    """Returns content of given requirements file."""
    with open(os.path.join(path)) as f:
        return [
            line.rstrip() for line in f if not (line.isspace() or line.startswith("#"))
        ]


setup(
    name="IPRO",
    version="0.1",
    description="Iterated Pareto Referent Optimisation",
    packages=["ipro"],
    author="Willem Röpke",
    install_requires=_parse_requirements("requirements.txt"),
    python_requires=">=3.10",
)
