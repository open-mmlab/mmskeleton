import os
import sys
import platform
import subprocess
import time

from setuptools import find_packages, setup, Extension
from setuptools.command.install import install

import numpy as np
from Cython.Build import cythonize  # noqa: E402
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class torch_and_cython_build_ext(BuildExtension):
    def finalize_options(self):
        if self.distribution.ext_modules:
            nthreads = getattr(self, 'parallel', None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            from Cython.Build.Dependencies import cythonize
            self.distribution.ext_modules[:] = cythonize(
                self.distribution.ext_modules, nthreads=nthreads, force=self.force)
        super().finalize_options()


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 7
PATCH = ''
SUFFIX = 'rc1'
SHORT_VERSION = '{}.{}.{}{}'.format(MAJOR, MINOR, PATCH, SUFFIX)

version_file = 'mmskeleton/version.py'


def get_git_hash():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH', 'HOME']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        sha = out.strip().decode('ascii')
    except OSError:
        sha = 'unknown'

    return sha


def get_hash():
    if os.path.exists('.git'):
        sha = get_git_hash()[:7]
    elif os.path.exists(version_file):
        try:
            from mmskeleton.version import __version__
            sha = __version__.split('+')[-1]
        except ImportError:
            raise ImportError('Unable to get git version')
    else:
        sha = 'unknown'

    return sha


def write_version_py():
    content = """# GENERATED VERSION FILE
# TIME: {}
__version__ = '{}'
short_version = '{}'
mmskl_home = r'{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha
    MMSKELETON_HOME = os.path.dirname(os.path.realpath(__file__))

    with open(version_file, 'w') as f:
        f.write(
            content.format(time.asctime(), VERSION, SHORT_VERSION,
                           MMSKELETON_HOME))


def get_version():
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


def get_requirements(filename='requirements.txt'):
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, filename), 'r') as f:
        requires = [line.replace('\n', '') for line in f.readlines()]
    return requires


def make_cython_ext(name, module, sources):
    extra_compile_args = None
    if platform.system() != 'Windows':
        extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings']
        }
    extension = Extension('{}.{}'.format(
        module, name), [os.path.join(*module.split('.'), p) for p in sources],
                          include_dirs=[np.get_include()],
                          language='c++',
                          extra_compile_args=extra_compile_args)
    extension, = cythonize(extension)
    return extension


def make_cuda_ext(name, module, sources, include_dirs=[]):

    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError('CUDA is required to compile MMSkeleton!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        include_dirs=include_dirs,
        extra_compile_args={
            'cxx': [],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':

    install_requires = get_requirements()
    if "--mmdet" in sys.argv:
        sys.argv.remove("--mmdet")
        install_requires += ['mmdet']

        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'https://github.com/open-mmlab/mmdetection/archive/v1.0rc1.zip', '-v'])

    write_version_py()
    setup(
        name='mmskeleton',
        version=get_version(),
        scripts=['./tools/mmskl'],
        description='Open MMLab Skeleton-based Human Understanding Toolbox',
        long_description=readme(),
        keywords='computer vision, human understanding, action recognition',
        url='https://github.com/open-mmlab/mmskeleton',
        packages=find_packages(exclude=('configs', 'tools', 'demo')),
        package_data={'mmskeleton.ops': ['*/*.so']},
        classifiers=[
            'Development Status :: 4 - Beta',
            'License :: OSI Approved :: Apache Software License',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
        ],
        license='Apache License 2.0',
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        dependency_links=[
            'https://github.com/open-mmlab/mmdetection/tarball/v1.0rc1/#egg=mmdet-v1.0rc1'
        ],
        install_requires=install_requires,
        ext_modules=[
            make_cython_ext(name='cpu_nms',
                            module='mmskeleton.ops.nms',
                            sources=['cpu_nms.pyx']),
            make_cuda_ext(name='gpu_nms',
                          module='mmskeleton.ops.nms',
                          sources=['nms_kernel.cu', 'gpu_nms.pyx'],
                          include_dirs=[np.get_include()]),
        ],
        cmdclass={
            'build_ext': torch_and_cython_build_ext,
        },
        zip_safe=False)
