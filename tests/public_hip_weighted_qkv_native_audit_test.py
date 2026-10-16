"""Weighted QKV-specific static call contract and shared text-preservation negative controls."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_public_hip_weighted_qkv_native import validate_calls, PUBLIC, IMPORTS, audit_text


def fixture():
    names=['_start','_init','main','ps5_fp_ieee','ps5hip_install_shell_exit_probe',
           'ps5hip_install_vector_title_finalizer','_Zdump_exclusive','_ZfinalizedEv',
           '_Zwatchdog_start','_Zwatchdog_finish','_Zfinite_floats','_Zq8_domain','_Znan_seed',*PUBLIC]
    values={n:[0x100+i*0x100] for i,n in enumerate(names)}
    slots={n:{0x8000+i*8} for i,n in enumerate(IMPORTS)}
    call=lambda n:f' callq 0x{values[n][0]:x} <{n}>\n'
    imported=lambda n:f' callq *0x8(%rip) # 0x{next(iter(slots[n])):x}\n'
    bodies={v[0]:' retq\n' for v in values.values()}
    bodies[values['_start'][0]]=call('ps5_fp_ieee')+imported('_init_env')+''.join(call(n) for n in ['ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer','_init','main'])
    bodies[values['main'][0]]=call('_Zwatchdog_start')+''.join(call(n) for n in PUBLIC)+call('hipModuleLaunchKernel')*2+call('hipMemcpy')*3+call('_Zfinite_floats')*4+call('_Zq8_domain')+call('_Znan_seed')*3+call('_Zdump_exclusive')+call('_Zwatchdog_finish')
    bodies[values['ps5hip_install_vector_title_finalizer'][0]]=f' leaq 0x0(%rip), %rax # 0x{values["_ZfinalizedEv"][0]:x} <_ZfinalizedEv>\n'+imported('atexit')
    bodies[values['_Zdump_exclusive'][0]]=''.join(imported(n) for n in ('open','write','fsync','close'))
    return values,bodies,slots,lambda address,size:(b'\0'*size,None)


class NativeWeightedQkvAudit(unittest.TestCase):
    def test_valid_exact_weighted_qkv_contract(self):
        result=validate_calls(*fixture())
        self.assertEqual(len(result['functions']),21)
        self.assertEqual(result['dump_helper'],'_Zdump_exclusive')

    def test_missing_duplicate_or_false_calls(self):
        for target,old,new in [('main','<hipMalloc>','<wrong>'),('main','<_Zdump_exclusive>','<wrong>'),
                               ('main','callq','leaq'),('_start','<_init>','<wrong>')]:
            values,bodies,slots,reader=fixture()
            bodies[values[target][0]]=bodies[values[target][0]].replace(old,new)
            with self.subTest(target=target,old=old),self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)
        for name in ('hipModuleLaunchKernel','hipMemcpy','_Zfinite_floats','_Zq8_domain','_Znan_seed','_Zdump_exclusive','_Zwatchdog_finish'):
            values,bodies,slots,reader=fixture()
            bodies[values['main'][0]]+=f' callq 0x{values[name][0]:x} <{name}>\n'
            with self.subTest(duplicate=name),self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)

    def test_actual_import_and_finalizer_binding(self):
        for target in ('_Zdump_exclusive','ps5hip_install_vector_title_finalizer','_start'):
            values,bodies,slots,reader=fixture()
            bodies[values[target][0]]=bodies[values[target][0]].replace('# 0x8','# 0x9')
            with self.subTest(target=target),self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)
        values,bodies,slots,reader=fixture()
        bodies[values['ps5hip_install_vector_title_finalizer'][0]]=bodies[values['ps5hip_install_vector_title_finalizer'][0]].replace(f'0x{values["_ZfinalizedEv"][0]:x}','0xdead')
        with self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)

    def test_dump_sync_order_and_ambiguous_helper(self):
        values,bodies,slots,reader=fixture()
        address=values['_Zdump_exclusive'][0]
        bodies[address]='\n'.join(reversed(bodies[address].splitlines()))
        with self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)
        values,bodies,slots,reader=fixture();values['_Zother_dump_exclusive']=[0x9000]
        with self.assertRaises(Exception):validate_calls(values,bodies,slots,reader)

    def test_shared_text_permissions_and_byte_identity(self):
        def reader(data,flags):return lambda address,size:(data,(0,flags))
        readers=[reader(b'abc',5),reader(b'abc',1),reader(b'abc',1)]
        audit_text(readers,{0x100},{0x100:3})
        for index,data,flags in [(0,b'abc',7),(1,b'abc',5),(2,b'abd',1)]:
            altered=readers[:];altered[index]=reader(data,flags)
            with self.subTest(index=index),self.assertRaises(Exception):audit_text(altered,{0x100},{0x100:3})
        with self.assertRaises(Exception):audit_text(readers,{0x100},{})

if __name__=='__main__':unittest.main()
