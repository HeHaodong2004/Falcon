"""Small NVRTC loader using Torch's allocator, device and current stream."""
import ctypes as ct
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import torch
from eventwm_graph.stage2_segment.cuda_kernel import checked


class CudaModule:
    def __init__(self, source, device):
        self.device = torch.device(device)
        self.functions = {}
        self.driver = ct.CDLL('libcuda.so.1')
        with torch.cuda.device(self.device):
            torch.empty(1, device=self.device)
            cap = torch.cuda.get_device_capability(self.device)
            options = [('--gpu-architecture=compute_%d%d'%cap).encode(), b'--fmad=false', b'--prec-div=true', b'--prec-sqrt=true', b'--std=c++11']
            self.identity = hashlib.sha256(source.encode()+b'\0'.join(options)).hexdigest()
            cache = Path(os.environ['NUMBA_CACHE_DIR'])/'geometry_cuda'
            cache.mkdir(parents=True, exist_ok=True)
            path = cache/(self.identity+'.ptx')
            if not path.exists():
                lib = ct.CDLL(str(Path(sys.prefix)/'lib/libnvrtc.so'))
                program = ct.c_void_p()
                checked(lib, 'nvrtcCreateProgram', [ct.POINTER(ct.c_void_p), ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_void_p, ct.c_void_p], ct.byref(program), source.encode(), b'geometry.cu', 0, None, None)
                try:
                    args = (ct.c_char_p*len(options))(*options)
                    try:
                        checked(lib, 'nvrtcCompileProgram', [ct.c_void_p, ct.c_int, ct.POINTER(ct.c_char_p)], program, len(options), args)
                    except RuntimeError:
                        size = ct.c_size_t()
                        checked(lib, 'nvrtcGetProgramLogSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
                        buf = ct.create_string_buffer(size.value)
                        checked(lib, 'nvrtcGetProgramLog', [ct.c_void_p, ct.c_void_p], program, buf)
                        raise RuntimeError(buf.value.decode())
                    size = ct.c_size_t()
                    checked(lib, 'nvrtcGetPTXSize', [ct.c_void_p, ct.POINTER(ct.c_size_t)], program, ct.byref(size))
                    buf = ct.create_string_buffer(size.value)
                    checked(lib, 'nvrtcGetPTX', [ct.c_void_p, ct.c_void_p], program, buf)
                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                        stream.write(buf.raw)
                        temporary = stream.name
                    os.replace(temporary, path)
                finally:
                    checked(lib, 'nvrtcDestroyProgram', [ct.POINTER(ct.c_void_p)], ct.byref(program))
            self.module = ct.c_void_p()
            checked(self.driver, 'cuModuleLoadData', [ct.POINTER(ct.c_void_p), ct.c_void_p], ct.byref(self.module), ct.create_string_buffer(path.read_bytes()))

    def launch(self, name, total, values, threads=128):
        if not total:
            return
        with torch.cuda.device(self.device):
            if name not in self.functions:
                function = ct.c_void_p()
                checked(self.driver, 'cuModuleGetFunction', [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_char_p], ct.byref(function), self.module, name.encode())
                self.functions[name] = function
            args = [ct.c_void_p(v.data_ptr()) if isinstance(v, torch.Tensor) else v for v in values]
            pointers = (ct.c_void_p*len(args))(*[ct.addressof(v) for v in args])
            stream = ct.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)
            checked(self.driver, 'cuLaunchKernel', [ct.c_void_p, *([ct.c_uint]*7), ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_void_p], self.functions[name], (total+threads-1)//threads, 1, 1, threads, 1, 1, 0, stream, pointers, None)
