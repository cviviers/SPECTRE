from pathlib import Path
from setuptools import setup

def read_requirements():
    req = Path(__file__).with_name("requirements.txt")
    if req.exists():
        lines = [r.strip() for r in req.read_text().splitlines()]
        return [r for r in lines if r and not r.startswith("#")]
    return []

setup(install_requires=read_requirements())