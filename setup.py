from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy 


setup(
    name='darknet_weights',
    ext_modules = [
        Extension(
            "darknet.weights_reader.weights",
            sources=["darknet/weights_reader/weights.pyx" #,"darknet_reader/weights_loader.cpp"
            ],
            language="c++",
            include_dirs=['./darknet/weights_reader']
    )],
    include_dirs=['darknet/weights_reader', numpy.get_include()],
    cmdclass={ 'build_ext': build_ext },
    packages=['darknet_weights'],
    package_dir={'darknet_weights':'darknet/weights_reader'},
)
