"""Build a receipt only; model weights and authenticated modules remain external."""
import argparse,json
from pathlib import Path
from public_hip_layer_assets import assemble,layer_inputs


def build(parents,reference,output):
 data=assemble(parents,reference)
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 (output/'receipt.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 return layer_inputs(output)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('getrows','normalized','weighted','q4','layer','reference','output'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();result=build({n:getattr(a,n) for n in ('getrows','normalized','weighted','q4','layer')},a.reference,a.output)
 print(json.dumps({'receipt':result['receipt'],'receipt_sha256':result['receipt_sha256'],'assets':result['assets']},indent=2))


if __name__=='__main__':main()
