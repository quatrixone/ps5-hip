from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent
src=p/'stress-inputs/late-callee-save.ll';dst=p/'stress-inputs/late-callee-unbounded.ll';assert not dst.exists()
dst.write_text(src.read_text().replace('"amdgpu-num-vgpr"="40"',''))
(p/'late-callee-unbounded-input.json').write_text(json.dumps({'file':str(dst.relative_to(p)),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'derived_from_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'change':'Remove num-vgpr40 stress limit; preserve source.'},indent=2)+'\n')
