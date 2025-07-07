from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="diffusers-lacon",
    version="0.1.0",
    author="LaCon Diffusers Implementation",
    author_email="",
    description="LaCon (Late-Constraint Diffusion) implementation for diffusers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/diffusers-lacon",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
            "mypy",
        ],
        "wandb": ["wandb>=0.13.0"],
        "xformers": ["xformers>=0.0.16"],
    },
    entry_points={
        "console_scripts": [
            "lacon-train=diffusers_lacon.training.train_condition_aligner:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)