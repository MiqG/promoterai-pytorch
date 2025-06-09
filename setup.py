from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="promoterai-pytorch",
    version="0.1.0",
    packages=["promoterai_pytorch"],
    author="Miquel Anglada Girotto",
    url="https://github.com/MiqG/promoterai-pytorch",
    description="PromoterAI model in pytorch.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        "torch>=1.9",
        "transformers>=4.0.0"
    ],
    python_requires=">=3.7"
)