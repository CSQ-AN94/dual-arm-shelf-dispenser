from glob import glob

from setuptools import setup

package_name = "grabber_robot_state_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Grabber",
    maintainer_email="csq15705100218@gmail.com",
    description="Read-only RealMan SDK to /joint_states bridge for the dual RM75.",
    license="MIT",
    entry_points={
        "console_scripts": [
            f"joint_state_bridge = {package_name}.joint_state_bridge:main",
            f"verify_current_state = {package_name}.verify_current_state:main",
            f"dump_joint_mapping = {package_name}.dump_joint_mapping:main",
        ],
    },
)
