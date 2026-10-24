#!/usr/bin/env python3
"""Explicit reproducible build of the exact Habitat native batch shim."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig
import tempfile


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo))
    from eventwm_graph.partial.native_batch import DEFAULT_BUILD,HABITAT_SHA256
    import habitat_sim._ext.habitat_sim_bindings as native
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_BUILD)
    parser.add_argument('--compiler',default=os.environ.get('CXX','g++'))
    args=parser.parse_args()
    if platform.machine()!='x86_64' or sys.version_info[:2]!=(3,9):
        raise SystemExit('Only the verified x86_64 CPython 3.9 Habitat build is supported')
    library=Path(native.__file__).resolve()
    if digest(library)!=HABITAT_SHA256:
        raise SystemExit('Unsupported Habitat binary: ABI review and parity are required')
    # The existing torch package ships pybind11 2.6.2 headers. Habitat uses
    # pybind11 internals v4 and GCC cxxabi1011, which the build explicitly binds.
    torch_spec=importlib.util.find_spec('torch')
    includes=Path(torch_spec.submodule_search_locations[0])/'include'
    if not (includes/'pybind11/pybind11.h').exists():
        raise SystemExit('Existing torch pybind11 headers not found')
    source=repo/'eventwm_graph/partial/native_batch.cpp'
    directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=True)
    extension='_eventwm_native_batch'+sysconfig.get_config_var('EXT_SUFFIX')
    with tempfile.TemporaryDirectory(prefix='native_batch_build_',dir=directory) as temporary:
        target=Path(temporary)/extension
        command=[args.compiler,'-O3','-DNDEBUG','-shared','-std=c++14','-fPIC',
                 '-fvisibility=hidden','-DPYBIND11_BUILD_ABI="_cxxabi1011"',
                 '-I'+str(includes),'-I'+str(sysconfig.get_paths()['include']),
                 str(source),'-o',str(target),'-ldl']
        subprocess.run(command,check=True)
        os.replace(target,directory/extension)
    metadata={'schema':'eventwm_native_batch_build_1','source':str(source.relative_to(repo)),
              'source_sha256':digest(source),'habitat_library':str(library),
              'habitat_sha256':HABITAT_SHA256,'extension':extension,
              'extension_sha256':digest(directory/extension),'python':sys.version,
              'compiler':subprocess.check_output([args.compiler,'--version'],text=True).splitlines()[0],
              'command':shlex.join(command),'pybind_headers':str(includes/'pybind11'),
              'pybind_common_sha256':digest(includes/'pybind11/detail/common.h'),
              'pybind_internals_sha256':digest(includes/'pybind11/detail/internals.h')}
    (directory/'build.json').write_text(json.dumps(metadata,indent=2,sort_keys=True)+'\n')
    print(json.dumps(metadata,indent=2,sort_keys=True))


if __name__=='__main__':main()
