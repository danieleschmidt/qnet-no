from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qnet-no",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.ai",
    description="Quantum-Network Neural Operator Library for distributed quantum computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/qnet-no",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "tensornetwork>=0.4.6",
        "optax>=0.1.4",
        "flax>=0.6.0",
        "pennylane>=0.32.0",
        "qiskit>=0.45.0",
        "strawberryfields>=0.21.0",
        "networkx>=2.8.0",
        "pydantic>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=1.0.0",
            "flake8>=5.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "gpu": [
            "cupy-cuda11x>=11.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qnet-no=qnet_no.cli:main",
        ],
    },
)