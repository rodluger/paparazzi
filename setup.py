"""Install script for `paparazzi`."""
import sys
import glob
from setuptools import setup

if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__PAPARAZZI_SETUP__ = True
import paparazzi as pp


setup(
    name="paparazzi",
    version=pp.__version__,
    author="Rodrigo Luger",
    author_email="rodluger@gmail.com",
    url="https://github.com/rodluger/paparazzi",
    description="Photos of the stars",
    license="GPL",
    packages=["paparazzi"],
    install_requires=["scipy>=1.2.1", "starry>=1.0.0", "celerite"],
    data_files=[("maps", glob.glob("paparazzi/*.jpg"))],
    include_package_data=True,
    zip_safe=False,
)
