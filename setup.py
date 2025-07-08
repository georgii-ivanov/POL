from setuptools import setup, find_packages

setup(
    name="proof-of-learning-blockchain",
    version="1.0.0",
    description="Proof of Learning consensus blockchain with distributed GPT AI training",
    author="POL Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt").readlines()
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "pol-node=pol.cli:main",
            "pol-miner=pol.miner:main",
            "pol-api=pol.api:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 