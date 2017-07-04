from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy 


setup(
    name='darknet_weights',
    ext_modules = [
        Extension(
            "darknet.weights_reader.weights",
            sources=[
                "darknet/weights_reader/weights.pyx" #,"darknet_reader/weights_loader.cpp"
            ],
            language="c++",
            include_dirs=['./darknet/weights_reader']
            ), 
        Extension(
            "darknet.network.box_operations",
            sources=[
                "darknet/network/box.c",
                "darknet/network/box_operations.pyx"
            ],
            #language="c++",
            include_dirs=["./darknet/network"]
        )],
    include_dirs=[
        'darknet/weights_reader', 
        numpy.get_include(),
        'darknet/network'
    ],
    cmdclass={ 'build_ext': build_ext },
    packages=[
        'darknet_weights', 
        'darknet_boxes'
    ],
    package_dir={
        'darknet_weights':'darknet/weights_reader',
        'darknet_boxes': 'darknet/network'
    },
)
