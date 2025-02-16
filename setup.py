from setuptools import setup, find_packages

setup(
    name='thevisiblehand',
    version='0.1.3',    
    description='A Python package to mask hands in videos using Segment Anything Model 2 (SAM2) and Mediapipe.',
    url='https://github.com/abhishekk962/thevisiblehand',
    author='Abhishek Kumar',
    author_email='abhishekk962@gmail.com',
    license='MIT',
    ppackages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'mediapipe',
        'numpy',
        'imageio',
        'matplotlib',
        'Pillow',
        'torch',
        'torchvision',
        'opencv-python',
        'scipy',
        'sam-2 @ git+https://github.com/facebookresearch/sam2.git'
    ],
    entry_points = {
        'console_scripts': [
            'thevisiblehand = thevisiblehand.cli:main',
        ],
    }
)