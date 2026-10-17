"""Run the shared payload/final-authentication invariants for model-layer."""
import sys,types,unittest
from unittest.mock import patch
import source_prefix_authentication_test as prefix

class LayerAuthentication(prefix.Authentication):
 def setUp(self):
  super().setUp()
  self.obj.manifest['llm_client']['phase']='model-layer';self.save()
 def test_prefix_imports_are_primed(self):
  names=('source_llm_layer_log','verify_llm_layer_readbacks','build_llm_layer_reference','public_hip_layer_assets')
  modules={n:types.ModuleType(n) for n in names}
  with patch.dict(sys.modules,modules):self.obj.preload_prefix_validators()

if __name__=='__main__':unittest.main()
