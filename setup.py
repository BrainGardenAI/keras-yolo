from distutils.core import setup, Extension
from Cython.Distutils import build_ext

setup(
    name='darknet_weights',
    ext_modules = [
        Extension(
            "darknet_reader.weights",
            sources=["darknet_reader/weights.pyx" #,"darknet_reader/weights_loader.cpp"
            ],
            language="c++",
            include_dirs=['./darknet_reader']
    )],
    include_dirs=['darknet_reader'],
    cmdclass={ 'build_ext': build_ext },
    packages=['darknet_weights'],
    package_dir={'darknet_weights':'darknet_reader'},
)
