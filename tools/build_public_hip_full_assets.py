"""Build a receipt only; model weights and authenticated modules remain external."""
import argparse,json
from pathlib import Path
from public_hip_full_assets import assemble,full_inputs


def build(parents,reference,output,profile='smoke'):
 data=assemble(parents,reference,profile)
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
 (output/'receipt.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
 return full_inputs(output)


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('getrows','normalized','weighted','q4','layer','reference','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--profile',choices=('smoke','generation','corpus','extended'),default='smoke')
 a=p.parse_args();result=build({n:getattr(a,n) for n in ('getrows','normalized','weighted','q4','layer')},a.reference,a.output,a.profile)
 print(json.dumps({'receipt':result['receipt'],'receipt_sha256':result['receipt_sha256'],'assets':result['assets']},indent=2))


if __name__=='__main__':main()
