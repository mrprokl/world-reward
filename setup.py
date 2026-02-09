from setuptools import find_packages, setup

setup(
    name="worldreward",
    version="0.1.0",
    description="Physics-verifiable evaluation pipeline for video/3D world models.",
    python_requires=">=3.11",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"worldreward": ["builtin_configs/*.yaml"]},
    install_requires=[
        "google-genai>=1.0.0",
        "prompt_toolkit>=3.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "mypy>=1.12.0",
            "pytest>=8.0.0",
            "ruff>=0.8.0",
            "types-PyYAML>=6.0.12.20240917",
        ]
    },
    entry_points={"console_scripts": ["worldreward=worldreward.main:main"]},
)
