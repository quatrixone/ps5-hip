"""Optional, isolated compilation cache for the existing PS5 compiler wrapper."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def shared_libraries(tool):
    listing = subprocess.check_output(['/usr/bin/ldd', tool], text=True, timeout=30,
                                      env={'PATH': '/usr/bin:/bin', 'LC_ALL': 'C'})
    dependencies = {Path('/usr/bin/ldd')}
    for line in listing.splitlines():
        fields = line.split()
        if 'not found' in line:
            raise ValueError('ccache shared library missing: ' + line.strip())
        name = fields[2] if len(fields) > 2 and fields[1] == '=>' else fields[0] if fields else ''
        if name.startswith('/'):
            path = Path(name)
            if not path.is_file():
                raise ValueError('ccache dependency is not a file: ' + name)
            dependencies.add(path)
        elif fields and not name.startswith('linux-vdso'):
            raise ValueError('unrecognized ccache dependency: ' + line.strip())
    if len(dependencies) == 1:
        raise ValueError('ccache shared dependency inventory is empty')
    return sorted(dependencies)


def configuration(root, env, identities=True):
    mode = env.get('PS5_HIP_CCACHE', 'auto')
    if mode not in ('auto', '0', '1'):
        raise ValueError('PS5_HIP_CCACHE must be auto, 0 or 1')
    tool = shutil.which('ccache', path='/usr/bin:/bin') if mode != '0' else None
    if mode == '1' and tool is None:
        raise ValueError('PS5_HIP_CCACHE=1 requires ccache')
    if tool is None:
        return {'enabled': False, 'requested': mode}, []
    llvm = Path('/usr/bin/llvm-config-18')
    compiler = Path(subprocess.check_output([str(llvm), '--bindir'], text=True).strip()) / 'clang++'
    files = [compiler.resolve(), llvm.resolve(), root / 'tools/ps5-clangxx', root / 'tools/native_ccache.py',
             root / '.deps/sdk/bin/prospero-clang++', root / '.deps/sdk/bin/clang++',
             root / '.deps/sdk/bin/prospero-llvm-config']
    cache = Path(env.get('PS5_HIP_CCACHE_DIR', str(Path.home() / '.cache/ps5-hip/ccache'))).expanduser().absolute()
    # Command-line configuration overrides even a cache-local configuration file.
    options = ['compiler_type=clang', 'compiler_check=content', 'direct_mode=false',
               'sloppiness=', 'remote_storage=', 'namespace=ps5-hip-native-v1',
               'max_size=1GiB', 'hash_dir=true', 'hard_link=false',
               'extra_files_to_hash=' + ':'.join(map(str, files)), 'cache_dir=' + str(cache)]
    record = {'enabled': True, 'requested': mode, 'tool': str(Path(tool).resolve()),
              'directory': str(cache), 'options': options}
    if identities:
        record.update(tool_sha256=sha(tool), inputs={str(p): sha(p) for p in [*files, *shared_libraries(tool)]})
    return record, [tool, *options]


def verify(root, record):
    if not isinstance(record, dict) or record.get('requested') not in ('auto', '0', '1'):
        raise ValueError('missing or invalid compiler cache record')
    if record.get('enabled') is False:
        if record not in ({'enabled': False, 'requested': '0'}, {'enabled': False, 'requested': 'auto'}):
            raise ValueError('invalid disabled compiler cache record')
        return
    if record.get('enabled') is not True or record['requested'] == '0':
        raise ValueError('invalid enabled compiler cache record')
    current, _ = configuration(root, {'PS5_HIP_CCACHE': '1', 'PS5_HIP_CCACHE_DIR': record.get('directory', '')})
    current['requested'] = record['requested']
    if current != record:
        raise ValueError('compiler cache identity or safe configuration changed')


def bound_inputs(record):
    return ({**record['inputs'], record['tool']: record['tool_sha256']}
            if record['enabled'] else {})


def invocation(root, argv, env):
    clean = {k: v for k, v in env.items() if not k.startswith('CCACHE_')}
    clean['CCACHE_CONFIGPATH'] = '/dev/null'
    if '-c' not in argv or any(option in argv for option in ('-M', '-MM', '-E', '-###')):
        return argv, clean
    _, prefix = configuration(root, clean, identities=False)
    return [*prefix, *argv], clean


def main():
    root = Path(__file__).resolve().parents[1]
    argv, env = invocation(root, sys.argv[1:], os.environ)
    os.execvpe(argv[0], argv, env)


if __name__ == '__main__':
    main()
