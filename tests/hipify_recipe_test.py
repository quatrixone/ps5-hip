import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from probe_hipify_client import check_translation

class TranslationBoundary(unittest.TestCase):
    def setUp(self):
        self.original='#include <cuda_runtime.h>\nvoid f(){ cudaMalloc(0,1); k<<<(n+63)/64, 64>>>(a); }\n'
        self.translated='#include "hip/hip_runtime.h"\n#include <hip/hip_runtime.h>\nvoid f(){ hipMalloc(0,1); k<<<(n+63)/64, 64>>>(a); }\n'

    def test_only_declared_translation(self):
        counts=check_translation(self.original,self.translated)
        self.assertEqual(counts['cudaMalloc'],1)
        self.assertEqual(sum(counts.values()),1)

    def test_kernel_or_host_rewrite_rejected(self):
        for old,new in [('(n+63)/64','1'),('hipMalloc(0,1)','hipMalloc(0,2)'),('hipMalloc','cudaMalloc'),('#include "hip/hip_runtime.h"\n','')]:
            with self.subTest(change=new),self.assertRaises(ValueError):
                check_translation(self.original,self.translated.replace(old,new))

if __name__=='__main__':unittest.main()
