from setuptools import setup, find_namespace_packages

setup(
    name='vsdkx-model-bayesian',
    url='https://github.com/natix-io/vsdkx-model-bayesian.git',
    author='Helmut',
    author_email='helmut@natix.io',
    namespace_packages=['vsdkx', 'vsdkx.model'],
    packages=find_namespace_packages(include=['vsdkx*']),
    dependency_links=[
        'git+https://github.com/natix-io/vsdkx-core#egg=vsdkx-core'
    ],
    install_requires=[
        'vsdkx-core',
        'torch>=1.7.0',
        'opencv-python~=4.2.0.34',
        'torchvision>=0.8.1',
        'numpy==1.18.5',
    ],
    version='1.0',
)
