from setuptools import setup

setup(
    name='thevisiblehand',
    version='0.1.0',    
    description='A Python package to mask hands in videos using Segment Anything Model 2 (SAM2) and Mediapipe.',
    url='https://github.com/abhishekk962/thevisiblehand',
    author='Abhishek Kumar',
    author_email='abhishekk962@gmail.com',
    license='MIT',
    install_requires=[
        'mediapipe',
        'matplotlib',
        'opencv-python',
        'git+https://github.com/facebookresearch/sam2.git'
    ],
)