from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retinal-image-enhancement",
    version="1.0.0",
    author="Medical Image Enhancement Team",
    description="Zeiss Visuscout to Clarus retinal image enhancement system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/retinal-image-enhancement",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-model=training.train:main",
            "run-inference=training.inference:main",
            "start-backend=backend.main:start",
        ],
    },
)

