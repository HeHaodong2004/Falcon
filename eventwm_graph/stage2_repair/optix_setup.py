"""Prepare optional pinned build headers locally; never change the environment."""
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile
from .diagnose import OUT,sha,save

SDK_COMMIT='7b5c4e8608b8b4b601729f6240fc3fd53cb36d23'
WHEEL_SHA='92d04069a987e1fbc9213f8376d265df0f7bb42617d44f5eda1f496acea7f2d1'


def run():
    dest=OUT/'deps';dest.mkdir(exist_ok=True);sdk=dest/'optix-dev'
    if not sdk.exists():subprocess.run(['git','clone','--depth','1','--branch','v7.7.0','https://github.com/NVIDIA/optix-dev.git',str(sdk)],check=True)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=sdk,text=True).strip()
    if revision!=SDK_COMMIT:raise ValueError('unexpected external OptiX headers; use pinned v7.7.0')
    wheel=dest/'nvidia_cuda_runtime_cu11-11.8.89-py3-none-manylinux2014_x86_64.whl'
    if not wheel.exists():subprocess.run([sys.executable,'-m','pip','download','--no-deps','--dest',str(dest),'nvidia-cuda-runtime-cu11==11.8.89'],check=True)
    if sha(wheel)!=WHEEL_SHA:raise ValueError('CUDA header package checksum mismatch')
    target=dest/'cuda_runtime'
    if not(target/'nvidia/cuda_runtime/include/cuda.h').exists():
        with ZipFile(wheel)as archive:archive.extractall(target)
    receipt=dict(optix_repository='https://github.com/NVIDIA/optix-dev',optix_commit=revision,cuda_package=wheel.name,cuda_package_sha256=WHEEL_SHA,scope='External build headers and licenses under ignored output/deps only. No package installation, system/driver changes, or third-party header redistribution in source branch.')
    save(dest/'setup_receipt.json',receipt);print(receipt,flush=True)


if __name__=='__main__':run()
