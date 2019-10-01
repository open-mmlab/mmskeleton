import os
import platform
import subprocess
import time
import numpy as np
from setuptools import find_packages, setup, Extension
from Cython.Build import cythonize  # noqa: E402
from mmskeleton.ops.nms.setup_linux import custom_build_ext, CUDA


def readme():
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
    return content


MAJOR = 0
MINOR = 1
PATCH = ''
SUFFIX = 'rc0'
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
mmskl_home = '{}'
"""
    sha = get_hash()
    VERSION = SHORT_VERSION + '+' + sha
    MMSKELETON_HOME = os.path.dirname(os.path.realpath(__file__))

    with open(version_file, 'w') as f:
        f.write(content.format(time.asctime(), VERSION, SHORT_VERSION, MMSKELETON_HOME))


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


if __name__ == '__main__':
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
            'git+https://github.com/open-mmlab/mmdetection#egg=mmdet'
        ],
        install_requires=get_requirements(),
        ext_modules=[
            Extension("mmskeleton.ops.nms.cpu_nms",
                      ["mmskeleton/ops/nms/cpu_nms.pyx"],
                      extra_compile_args={
                          'gcc': ["-Wno-cpp", "-Wno-unused-function"]
                      },
                      include_dirs=[np.get_include()]),
            Extension(
                'mmskeleton.ops.nms.gpu_nms',
                [
                    'mmskeleton/ops/nms/nms_kernel.cu',
                    'mmskeleton/ops/nms/gpu_nms.pyx'
                ],
                library_dirs=[CUDA['lib64']],
                libraries=['cudart'],
                language='c++',
                runtime_library_dirs=[CUDA['lib64']],
                # this syntax is specific to this build system
                # we're only going to use certain compiler args with nvcc and not with
                # gcc the implementation of this trick is in customize_compiler() below
                extra_compile_args={
                    'gcc': ["-Wno-unused-function"],
                    'nvcc': [
                        '-arch=sm_35', '--ptxas-options=-v', '-c',
                        '--compiler-options', "'-fPIC'"
                    ]
                },
                include_dirs=[np.get_include(), CUDA['include']]),
        ],
        cmdclass={'build_ext': custom_build_ext},
        zip_safe=False)
