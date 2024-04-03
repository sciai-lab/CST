from setuptools import setup, find_packages, Extension
import os

# Function to discover C++ sources
def discover_cpp_sources(directory):
    cpp_sources = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp"):
                cpp_file_path = os.path.relpath(os.path.join(root, file))
                cpp_sources.append(cpp_file_path)
    return cpp_sources

# Function to create an Extension for each C++ source
def create_extensions(cpp_sources):
    extensions = []
    for cpp_source in cpp_sources:
        # Define the name of the extension
        module_name = os.path.splitext(os.path.basename(cpp_source))[0]
        # Define the extension
        extension = Extension(
            module_name,
            sources=[cpp_source],
            extra_compile_args=['-Wall', '-Werror', '-O3', '-fpic'],
            extra_link_args=['-shared'],
            language='c++'
        )
        extensions.append(extension)
    return extensions

# Discover C++ source files
#cpp_sources = discover_cpp_sources('lib/')
#print(cpp_sources)
# Create extensions
#extensions = create_extensions(cpp_sources)

# Function to load requirements from requirements.txt
def load_requirements(filename):
    with open(filename, 'r') as file:
        requirements = file.read().splitlines()
    return requirements

#check python version
import sys
if (3, 7)<=sys.version_info <= (3, 8):
    requirements_file='requirements_python3.7.txt'
elif (3, 8)<=sys.version_info <= (3, 9):
    requirements_file= 'requirements_python3.8.txt'
requirements_file='requirements.txt'

print(requirements_file)
import time
time.sleep(4)
# Setup function
setup(
    name='CST',
    version='0.1',
    description='(Branched) Central Spanning Tree',
    license="MIT",
    author='Enrique Fita Sanmartin',
    author_email='enrique.fita.sanmartin@iwr.uni-heidelberg.de',
    packages=find_packages('lib/'),
    package_dir={'': 'lib'},
    # Include the pre-compiled .so files as package data
    package_data={
        # If your .so files are in a package called 'mypackage' under 'lib/',
        # include them like this:
        'CST.methods.mSTreg.geometry_optimization.lib': ['*.so'],
    },
    install_requires=load_requirements(requirements_file),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: MIT License',
        'Development Status :: 5 - Production/Stable',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
    ],
    zip_safe=False,
)

