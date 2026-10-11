"""Probe the pinned TheRock target configuration; no compiler build or console access.

Requires the upstream target CMake file at the ignored reference path below.
Its digest is checked before CMake evaluates it.
"""
import hashlib,json,subprocess,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
rev='4d1b9e560fa23ea2340fcf17b423bb281ab29aae'
directory=root/'.deps'/('therock-reference-'+rev)
source=directory/'cmake/therock_amdgpu_targets.cmake'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='9804daea3e725fecea4bf73990e6a4e6bfbdd4885795725ef1c89c3f0703d76e'
rows=[]
with tempfile.TemporaryDirectory(prefix='ps5hip-therock-targets-') as temporary:
    work=Path(temporary)
    for target,custom,expected in [('gfx1030',False,0),('gfx1013',False,1),('gfx1013',True,0)]:
        modules=work/(target+str(custom));modules.mkdir()
        if custom:
            (modules/'therock_custom_amdgpu_targets.cmake').write_text(
                'therock_add_amdgpu_target(gfx1013 "ps5-hip compiler experiment; unqualified" FAMILY ps5-experimental)\n')
        probe=work/'probe.cmake'
        probe.write_text(f'''cmake_minimum_required(VERSION 3.25)
set(CMAKE_MODULE_PATH "{modules}")
include("{source}")
set(THEROCK_AMDGPU_TARGETS {target})
therock_validate_amdgpu_targets()
if(NOT THEROCK_AMDGPU_TARGETS STREQUAL "{target}")
  message(FATAL_ERROR "Target expansion mismatch")
endif()
''')
        result=subprocess.run(['cmake','-P',str(probe)],capture_output=True,text=True)
        assert (result.returncode==0)==(expected==0),result.stdout+result.stderr
        if expected:assert "Unknown AMDGPU target 'gfx1013'" in result.stderr
        rows.append({'target':target,'custom_target_hook':custom,'configuration_passed':result.returncode==0})
reference=json.loads((root/'evidence/offline-therock-comparison.json').read_text())
receipt={key:reference[key] for key in ('revision','files','compiler_submodule')}
receipt.update({'schema':1,'classification':'source and target-configuration comparison; TheRock compiler not built',
                'hardware_gate_passed':False,'configuration_probes':rows,
                'cmake_version':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0],
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'selected_bringup_route':'existing LLVM 18.1.8 standalone device compiler',
                'limitations':['Custom target registration is not compiler or library qualification',
                               'No binary/ISA comparison against a built TheRock compiler yet']})
(root/'evidence/offline-therock-comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(rows,indent=2))
