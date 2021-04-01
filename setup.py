from setuptools import setup
from glob import glob

package_name = 'ros2_deeplearn_twist'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/deeplearn', glob('deeplearn/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ANI717',
    maintainer_email='animesh.ani@live.com',
    description='Deep Learning Package to Publish Twist Message for Robot Running',
    license='MIT License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'execute = ros2_deeplearn_twist.deeplearn_twist_publish_function:main',
        ],
    },
)
