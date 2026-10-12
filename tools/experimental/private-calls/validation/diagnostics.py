"""Validate real compiler proof diagnostics; synthetic tests only test this parser."""
import re

def check_diagnostics(log, source, expected, required_present=()):
    entries = re.findall(r'^buffer-call-proof (inventory|result) (\S+) (\S+)$', log, re.M)
    assert entries, 'missing proof diagnostics'
    inventory, results = {}, {}
    results_started = False
    for kind, name, value in entries:
        if kind == 'inventory':
            assert not results_started, 'inventory after proof began'
            assert name not in inventory and value in ('present', 'missing'), 'invalid inventory'
            inventory[name] = value
        else:
            results_started = True
            assert name not in results and value in ('proven', 'rejected'), 'invalid result'
            assert inventory.get(name) == 'present', 'proof without machine function'
            results[name] = value
    definitions = set(re.findall(r'^define\b[^\n]*@([^\s(]+)\(', source, re.M))
    assert definitions <= inventory.keys(), 'source definition omitted from inventory'
    assert set(results) == {n for n, state in inventory.items() if state == 'present'}, 'missing proof result'
    for name in required_present:
        assert inventory.get(name) == 'present', ('reachable machine function missing', name)
    for name, value in expected.items():
        assert results.get(name) == value, ('unexpected proof result', name, results.get(name))
    return {'inventory': inventory, 'results': results, 'all_inventory_before_results': True}
