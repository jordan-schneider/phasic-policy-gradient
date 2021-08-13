from setuptools import find_packages, setup  # type: ignore

packages = find_packages()
package_data = {package: ["py.typed"] for package in packages}

setup(
    name="phasic-policy-gradient",
    packages=packages,
    version="0.0.1",
    package_data=package_data,
)
