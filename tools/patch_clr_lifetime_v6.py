"""Preserve v5 proof inputs while adding the observed queue-status race fix."""
from patch_clr_lifetime import patch as previous
def patch(source):
 changed=previous(source)
 p=source/'rocclr/platform/commandqueue.hpp';text=p.read_text()
 assert text.count('  bool isActive_;')==1
 p.write_text(text.replace('  bool isActive_;','  std::atomic<bool> isActive_;'))
 return changed+[p]
