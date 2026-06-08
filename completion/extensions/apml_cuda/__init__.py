"""
JIT-compile and load the APML sparse CUDA kernel.
Usage:
    from extensions.apml_cuda import load_apml_sparse
    apml_sparse = load_apml_sparse()
"""

import os

def load_apml_sparse():
    # Ensure ninja is on PATH (installed in pgst env but may not be on system PATH)
    env_ninja = os.path.join(os.path.dirname(os.path.dirname(os.__file__)), 'bin', 'ninja')
    if os.path.isfile(env_ninja):
        os.environ['PATH'] = env_ninja + ':' + os.environ.get('PATH', '')

    from torch.utils.cpp_extension import load
    ext_dir = os.path.dirname(os.path.abspath(__file__))
    return load(
        name='apml_sparse',
        sources=[
            os.path.join(ext_dir, 'apml_sparse.cpp'),
            os.path.join(ext_dir, 'apml_sparse_kernel.cu'),
        ],
        extra_cuda_cflags=[
            '--extended-lambda',
            '--expt-relaxed-constexpr',
            '-std=c++17',
            '-O2',
        ],
        extra_cflags=['-O2'],
        verbose=True,
    )
