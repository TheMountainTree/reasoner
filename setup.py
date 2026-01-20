from setuptools import find_packages, setup

package_name = 'reasoner'

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
    maintainer='mlin',
    maintainer_email='1473953987@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'reasoner_vis = reasoner.reasoner_node_vis:main',
            'rgb_process = reasoner.pics_process:main',
            'single_try_gdnsam = reasoner.single_test_entry:main'
        ],
    },
)
