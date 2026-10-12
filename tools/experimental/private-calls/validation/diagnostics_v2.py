"""Strict real compiler inventory check with bounded dead-internal elimination."""
import re

DEAD_ORDINARY_ORIGINALS = frozenset(('_Z9call_leafPjj', '_Z11call_middlePjj'))

def linked_functions(metadata):
    functions = set()
    for block in re.findall(r'(?ms)^  Symbol \{\n(.*?)^  \}', metadata):
        if not re.search(r'^    Type: Function \(0x2\)$', block, re.M):
            continue
        name = re.search(r'^    Name: (\S+) \(', block, re.M)
        assert name and not re.search(r'^    Section: Undefined', block, re.M), 'invalid linked function'
        assert name[1] not in functions, 'duplicate linked function'
        functions.add(name[1])
    assert functions, 'missing linked function inventory'
    return functions

def check_diagnostics(log, source, expected, required_present=(), *, linked_metadata,
                      baseline_metadata, allowed_dead=()):
    linked = linked_functions(linked_metadata)
    baseline = linked_functions(baseline_metadata)
    assert linked == baseline, 'current/baseline linked function inventory differs'
    allowed_dead = set(allowed_dead)
    assert allowed_dead <= DEAD_ORDINARY_ORIGINALS, 'unapproved dead-function exception'
    definitions = {}
    for match in re.finditer(r'(?ms)^define\b([^\n]*?)@([^\s(]+)\([^\n]*\n.*?^\}', source):
        assert match[2] not in definitions, 'duplicate source definition'
        definitions[match[2]] = match
    assert definitions, 'missing source definitions'
    assert set(definitions) - baseline == allowed_dead, 'source/linked omission is not exact dead set'
    assert baseline <= definitions.keys(), 'linked function absent from source'
    outside = source
    for name in allowed_dead:
        match = definitions[name]
        assert re.search(r'\binternal\b', match[1]), 'omitted function is not internal'
        outside = outside.replace(match[0], '')
    for name in allowed_dead:
        assert not re.search(r'@' + re.escape(name) + r'(?![-a-zA-Z$._0-9])', outside), 'omitted internal still referenced'
    inventory, results = {}, {}
    proof_started = False
    for line in log.splitlines():
        if line.startswith('buffer-call-proof register-use '):
            assert re.fullmatch(r'buffer-call-proof register-use \S+ explicit [01] with-mask [01]', line), 'invalid register-use record'
            proof_started = True
        if not (line.startswith('buffer-call-proof inventory') or line.startswith('buffer-call-proof result')):
            continue
        entry = re.fullmatch(r'buffer-call-proof (inventory|result) (\S+) (\S+)', line)
        assert entry, 'malformed inventory/result record'
        kind, name, value = entry.groups()
        if kind == 'inventory':
            assert not proof_started, 'inventory after proof began'
            assert name not in inventory and value in ('present', 'missing'), 'invalid inventory'
            inventory[name] = value
        else:
            proof_started = True
            assert name not in results and value in ('proven', 'rejected'), 'invalid result'
            assert inventory.get(name) == 'present', 'proof without machine function'
            results[name] = value
    assert inventory, 'missing proof diagnostics'
    present = {n for n, state in inventory.items() if state == 'present'}
    assert present == linked, 'machine/linked function inventory differs'
    assert set(definitions) - inventory.keys() == allowed_dead, 'source definition omitted from inventory'
    assert not allowed_dead.intersection(inventory), 'eliminated definition unexpectedly inventoried'
    assert set(results) == present, 'missing proof result'
    for name in required_present:
        assert inventory.get(name) == 'present', ('reachable machine function missing', name)
    for name, value in expected.items():
        assert results.get(name) == value, ('unexpected proof result', name, results.get(name))
    return {'inventory': inventory, 'results': results, 'all_inventory_before_results': True,
            'all_inventory_before_register_use': True, 'linked_functions': sorted(linked),
            'baseline_linked_functions': sorted(baseline), 'dead_source_internals': sorted(allowed_dead)}
