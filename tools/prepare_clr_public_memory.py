"""Prepare real upstream SVM/hipFree changes without modifying pinned suppliers."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil,subprocess
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=R/'results/clr-source-v7';O.mkdir(parents=True,exist_ok=False)
base=B/'.local/g5-clr-ps5-adapter/results/clr-source-v7';shutil.copytree(base/'source',O/'source');metadata=json.loads((base/'preparation.json').read_text());shutil.copyfile(base/'ps5-aot.patch',O/'ps5-aot.patch')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for path,z in metadata['changed_sources'].items():assert h(O/path)==z,path
for name in ['ps5-affinity.patch','ps5-executable-path.patch']:
 subprocess.run(['git','apply','--check',str(R/'ports/clr'/name)],cwd=O/'source',check=True)
 subprocess.run(['git','apply',str(R/'ports/clr'/name)],cwd=O/'source',check=True)
up=B/'results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c';hip=O/'source/hipamd/src/hip_memory.cpp';hip.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(up/'hipamd/src/hip_memory.cpp',hip)
patches=[];changed={}
for relative in ['rocclr/platform/memory.cpp','hipamd/src/hip_memory.cpp']:
 file=O/'source'/relative;before=file.read_text();after=before
 if relative.startswith('rocclr'):
  old='''  Add(ret_u, ret_u + size);
  return ret;''';new='''#if defined(WITH_PS5_DEVICE)
  try { Add(ret_u, ret_u + size); }
  catch (...) { context.svmFree(ret); return nullptr; }
#else
  Add(ret_u, ret_u + size);
#endif
  return ret;''';assert after.count(old)==1;after=after.replace(old,new)
  old='''  Remove(reinterpret_cast<uintptr_t>(ptr));
  context.svmFree(ptr);''';new='''#if defined(WITH_PS5_DEVICE)
  context.svmFree(ptr);
  // A failed or still-referenced device allocation remains live and tracked.
  if (MemObjMap::FindMemObj(ptr) == nullptr) Remove(reinterpret_cast<uintptr_t>(ptr));
#else
  Remove(reinterpret_cast<uintptr_t>(ptr));
  context.svmFree(ptr);
#endif''';assert after.count(old)==1;after=after.replace(old,new)
 else:
  start=after.index('hipError_t ihipFree(');end=after.index('// ================================================================================================',start);part=after[start:end]
  old='''  if (memory_object != nullptr) {
    // Wait on the device''';new='''  if (memory_object != nullptr) {
#if defined(WITH_PS5_DEVICE)
    if (offset != 0) return hipErrorInvalidValue;
#endif
    // Wait on the device''';assert part.count(old)==1;part=part.replace(old,new)
  old='''        amd::SvmBuffer::free(memory_object->getContext(), ptr);''';new=old+'''
#if defined(WITH_PS5_DEVICE)
        if (amd::MemObjMap::FindMemObj(ptr) != nullptr) return hipErrorUnknown;
#endif''';assert part.count(old)==1;part=part.replace(old,new);after=after[:start]+part+after[end:]
 file.write_text(after);patches.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative));changed[relative]={'before':hashlib.sha256(before.encode()).hexdigest(),'after':h(file)}
patch=R/'ports/clr/ps5-public-memory.patch';patch.write_text(''.join(patches))
metadata['changed_sources']['source/rocclr/os/os_posix.cpp']=h(O/'source/rocclr/os/os_posix.cpp')
for name in changed:metadata['changed_sources']['source/'+name]=h(O/'source'/name)
metadata['public_memory_patch_sha256']=h(patch);metadata['public_memory_preparer_sha256']=h(Path(__file__));(O/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n');(O/'public-memory-changes.json').write_text(json.dumps(changed,indent=2)+'\n')
print(h(O/'preparation.json'))
