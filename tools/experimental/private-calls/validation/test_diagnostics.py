from diagnostics import check_diagnostics
from pathlib import Path
import hashlib,json
source='define void @root() {\n}\ndefine void @leaf() {\n}\n'
good='buffer-call-proof inventory root present\nbuffer-call-proof inventory leaf present\nbuffer-call-proof inventory external missing\nbuffer-call-proof result root proven\nbuffer-call-proof result leaf proven\n'
expected={'root':'proven','leaf':'proven'}
check_diagnostics(good,source,expected,expected)
bad=[
 '',good.replace('inventory leaf present','inventory leaf missing'),
 good.replace('buffer-call-proof inventory leaf present\n',''),
 good.replace('buffer-call-proof result leaf proven\n',''),
 good.replace('result root proven','result root rejected'),
 good+'buffer-call-proof inventory late present\n',
 good+'buffer-call-proof result leaf proven\n',
 good.replace('buffer-call-proof result root proven\n','buffer-call-proof result root proven\nbuffer-call-proof result external proven\n'),
]
for log in bad:
 try:check_diagnostics(log,source,expected,expected)
 except AssertionError:pass
 else:raise AssertionError('diagnostic mutation accepted')
p=Path(__file__).resolve().parent
report={'parser_sha256':hashlib.sha256((p/'diagnostics.py').read_bytes()).hexdigest(),'valid_case':1,'rejected_mutations':len(bad),'actual_compiler_proof_executed':False}
(p/'diagnostic-parser-tests.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
