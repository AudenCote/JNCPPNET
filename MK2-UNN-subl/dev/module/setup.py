from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="cython_file", 
    sources=["cython_unn_class.pyx", "UNN_link.cpp"],
    extra_compile_args=["-std=c++11"],
    language="c++",
    )]

setup(
    name = 'cython_unn_class',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )