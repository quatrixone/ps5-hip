"""Prepare an isolated v7 descendant; never modifies the preserved source snapshot."""
from pathlib import Path
import argparse,hashlib,json,shutil,difflib
from patch_clr_lifetime_v7 import patch as patch_lifetime
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
proof=json.loads((B/'results/clr-native-combined-v7/preparation.json').read_text());h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
S=B/'results/clr-native-combined-v7'
for path,digest in proof['sources'].items():assert h(S/path)==digest,path
shutil.copytree(S/'source',O/'source')
dp=O/'source/rocclr/device/devprogram.cpp';text=dp.read_text();old='''  if (isLC()) {
    ret = loadLC();
  } else {
    ret = loadHSAIL();
  }''';new='''#if defined(WITH_PS5_DEVICE)
  // AOT only: the PS5 adapter validates and owns the exact executable image.
  ret = type() == TYPE_EXECUTABLE &&
      setKernels(const_cast<void*>(binary().first), binary().second,
                 BinaryFd().first, BinaryFd().second, BinaryURI());
#else
'''+old+'''
#endif''';assert text.count(old)==1;text=text.replace(old,new);dp.write_text(text)
pp=O/'source/rocclr/platform/program.cpp';text=pp.read_text();start=text.index('bool Program::load(');end=text.index('\nconst std::string& Program::kernelNames',start);part=text[start:end]
part=part.replace('  ScopedLock sl(&programLock_);','''  ScopedLock sl(&programLock_);
#if defined(WITH_PS5_DEVICE)
  // Symbols are referenced by captured kernels; never replace a published table.
  if (symbolTable_ != nullptr) return true;
#endif''')
old='''  return true;
}''';new='''#if defined(WITH_PS5_DEVICE)
  // Publish only a completely loaded AOT symbol table; no stack-owned kernels.
  std::unique_ptr<symbols_t> pending(new symbols_t);
  for (const auto& entry : devicePrograms_) {
    for (const auto& kernel : entry.second->kernels()) {
      if (!(*pending)[kernel.first].setDeviceKernel(*entry.first, kernel.second)) return false;
    }
  }
  delete symbolTable_;
  symbolTable_ = pending.release();
#endif
  return true;
}''';assert part.count(old)==1;text=text[:start]+part.replace(old,new)+text[end:];text='#include <memory>\n'+text
needle='''                                  size_t foffset, std::string uri) {
  if (image != NULL'''
replacement='''                                  size_t foffset, std::string uri) {
#if defined(WITH_PS5_DEVICE)
  ScopedLock ps5Lock(&programLock_);
  if (symbolTable_ != nullptr) return CL_INVALID_OPERATION;
  try {
    if (image == nullptr || length < 64) return CL_INVALID_BINARY;
    (void)ps5hip::inspect_code_object(static_cast<const uint8_t*>(image), length);
  } catch (...) { return CL_INVALID_BINARY; }
#endif
  if (image != NULL'''
assert text.count(needle)==1;text=text.replace(needle,replacement);text='#if defined(WITH_PS5_DEVICE)\n#include "ps5hip/code_object.h"\n#endif\n'+text;pp.write_text(text)
changed=[dp,pp]+patch_lifetime(O/'source')
patch=''
for q in changed:
 relative=q.relative_to(O/'source');old=(S/'source'/relative).read_text()
 patch+=''.join(difflib.unified_diff(old.splitlines(True),q.read_text().splitlines(True),fromfile='a/'+str(relative),tofile='b/'+str(relative)))
(O/'ps5-aot.patch').write_text(patch)
(O/'preparation.json').write_text(json.dumps({'base_preparation_sha256':h(S/'preparation.json'),'changed_sources':{str(q.relative_to(O)):h(q) for q in changed},'generator_sha256':h(Path(__file__)),'lifetime_generator_sha256':h(Path(__file__).with_name('patch_clr_lifetime.py')),'lifetime_v6_sha256':h(Path(__file__).with_name('patch_clr_lifetime_v6.py')),'lifetime_v7_sha256':h(Path(__file__).with_name('patch_clr_lifetime_v7.py'))},indent=2)+'\n')
print(O)
