from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os
import imp


if os.name =='nt' :
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["darkflow/cython_utils/nms.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
            sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
            #libraries=["m"] # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        # Extension("darkflow.cython_utils.cy_yolo_findboxes",
        #     sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
        #     #libraries=["m"] # Unix-like specific
        #     include_dirs=[numpy.get_include()]
        # )
    ]

elif os.name =='posix' :
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["darkflow/cython_utils/nms.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),        
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
            sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"], # Unix-like specific
            include_dirs=[numpy.get_include()]
        ),
        # Extension("darkflow.cython_utils.cy_yolo_findboxes",
        #     sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
        #     libraries=["m"], # Unix-like specific
        #     include_dirs=[numpy.get_include()]
        # )
    ]

else :
    ext_modules=[
        Extension("darkflow.cython_utils.nms",
            sources=["trainer/darkflow/cython_utils/nms.pyx"],
            libraries=["m"] # Unix-like specific
        ),        
        Extension("darkflow.cython_utils.cy_yolo2_findboxes",
            sources=["darkflow/cython_utils/cy_yolo2_findboxes.pyx"],
            libraries=["m"] # Unix-like specific
        ),
        # Extension("darkflow.cython_utils.cy_yolo_findboxes",
        #     sources=["darkflow/cython_utils/cy_yolo_findboxes.pyx"],
        #     libraries=["m"] # Unix-like specific
        # )
    ]

setup(
    version="1.0.0",
	name='darkflow',
    description='Darkflow',
    license='GPLv3',
    url='',
    packages = find_packages(),
    ext_modules = cythonize(ext_modules)
)