from setuptools import find_packages, setup

package_name = 'my_robot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='My robot battery monitor system',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
    'console_scripts': [
        'battery_publisher = my_robot.battery_publisher:main',
        'battery_subscriber = my_robot.battery_subscriber:main',
        'serial_teleop = my_robot.serial_teleop:main',
        'snelheid_teleop = my_robot.snelheid_teleop:main',
        'teleop = my_robot.teleop:main',
    ],
    },


)
