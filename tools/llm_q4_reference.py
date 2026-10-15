"""Offline selected Q8_1/Q4_0 numerical references; no GPU execution.

Q4 lane order is tied to fixture-v3 ISA: block=(32*wave+lane)//2,
stride64, low/high nibble indices8*(lane%2)..+7, mixed F32 FMA at
0x1f10/0x1f1c, LDS waves1,2,3 then XOR16,8,4,2,1. Half operands
are promoted exactly; there is no intermediate half FMA result.
"""
from fractions import Fraction
import math
import struct

U = Fraction(1, 1 << 24)
MIN_NORMAL = 2.0 ** -126
WIDTHS = (2048, 5632)


def bits(value):
    return struct.unpack('<I', struct.pack('<f', value))[0]


def from_bits(value):
    return struct.unpack('<f', struct.pack('<I', value))[0]


def finite(value):
    if not math.isfinite(value):
        raise ValueError('nonfinite value')
    return value


def f32(value):
    """Exact RN-even rounding, including a possible double-rounding midpoint.

    Binary64 conversion puts the exact rational within one F32 neighbour of
    the answer. Compare exact rational distances of all three candidates.
    The initial scope excludes overflow and nonzero subnormal results.
    """
    if isinstance(value, float) and value == 0:
        return value  # Preserve negative zero input.
    value = Fraction(value)
    sign = value < 0
    magnitude = abs(value)
    try:
        nearest = bits(float(magnitude))
    except (OverflowError, struct.error):
        raise ValueError('F32 overflow') from None
    if nearest >= 0x7f800000:
        raise ValueError('F32 overflow')
    candidates = range(max(0, nearest - 1), min(0x7f7fffff, nearest + 1) + 1)
    selected = min(candidates, key=lambda b: (abs(Fraction(from_bits(b)) - magnitude), b & 1))
    result = from_bits(selected)
    if (magnitude and not result) or (result and result < MIN_NORMAL):
        raise ValueError('F32 subnormal outside initial scope')
    return -result if sign else result


def add(a, b):
    if a == 0 and b == 0:
        return -0.0 if math.copysign(1, a) < 0 and math.copysign(1, b) < 0 else 0.0
    return f32(Fraction(a) + Fraction(b))


def fma(a, b, c):
    if (a == 0 or b == 0) and c == 0:
        return -0.0 if math.copysign(1, a)*math.copysign(1, b) < 0 and math.copysign(1, c) < 0 else 0.0
    return f32(Fraction(a) * Fraction(b) + Fraction(c))


def div(a, b):
    if not b:
        raise ValueError('zero divisor')
    return f32(Fraction(a) / Fraction(b))


def half(value, allow_zero=True):
    finite(value)
    try:
        packed = struct.pack('<e', value)
    except (OverflowError, struct.error):
        raise ValueError('binary16 overflow') from None
    result = struct.unpack('<e', packed)[0]
    # Selected FLOAT_MODE0xf0 preserves half input/output denormals. Every
    # binary16 subnormal promotes exactly to a NORMAL binary32 value.
    if not math.isfinite(result):
        raise ValueError('binary16 nonfinite')
    if value != 0 and result == 0:
        raise ValueError('binary16 underflow')
    if not allow_zero and result == 0:
        raise ValueError('zero binary16 scale')
    return result


def butterfly(values):
    if len(values) != 32:
        raise ValueError('wave32 required')
    values = list(values)
    for mask in (16, 8, 4, 2, 1):
        values = [add(v, values[lane ^ mask]) for lane, v in enumerate(values)]
    return values[0]


def round_away(value):
    finite(value)
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def quantize_q8(values, padded_width=None):
    """GPU-format reference, NOT the ggml CPU quantize_row_q8_1_ref format.

    d is rounded F32 max/127; integers use this d before its half conversion.
    s is half of the ORIGINAL F32 butterfly sum. Exact bytes are the planned
    GPU comparison contract. Corrected GPU division has yet to be validated
    against these IEEE results on hardware; no approximate reciprocal model.
    """
    values = [f32(finite(v)) for v in values]
    padded_width = len(values) if padded_width is None else padded_width
    if not values or padded_width < len(values) or padded_width % 32:
        raise ValueError('invalid padded width')
    values += [0.0] * (padded_width - len(values))
    output = bytearray()
    for start in range(0, padded_width, 32):
        block = values[start:start + 32]
        maximum = max(map(abs, block))
        d = div(maximum, 127) if maximum else 0.0
        hd = half(d, allow_zero=not maximum)
        hs = half(butterfly(block))
        qs = [round_away(div(x, d)) if d else 0 for x in block]
        if any(q < -127 or q > 127 for q in qs):
            raise ValueError('quantization overflow')
        output += struct.pack('<ee32b', hd, hs, *qs)
    return bytes(output)


def q4_block(scale, nibbles):
    if len(nibbles) != 32 or any(type(q) is not int or not 0 <= q <= 15 for q in nibbles):
        raise ValueError('32 unsigned nibbles required')
    return struct.pack('<e', half(scale)) + bytes(nibbles[i] | (nibbles[i + 16] << 4) for i in range(16))


def decode(q4, q8):
    if not q4 or len(q4) % 18 or len(q8) != len(q4) // 18 * 36:
        raise ValueError('exact matching block lengths required')
    blocks = []
    for i in range(len(q4) // 18):
        raw = q4[18*i:18*i+18]
        d4 = half(struct.unpack_from('<e', raw)[0])
        d8, s8, *qs = struct.unpack_from('<ee32b', q8, 36*i)
        half(d8); half(s8)
        if d8 < 0 or any(q == -128 for q in qs) or (d8 == 0 and (s8 != 0 or any(qs))):
            raise ValueError('invalid GPU Q8 block')
        ns = [b & 15 for b in raw[2:]] + [b >> 4 for b in raw[2:]]
        blocks.append((d4, d8, s8, ns, qs))
    return blocks


def semantic_fp64(q4, q8):
    """Independent full-block semantic oracle; original-sum correction retained.

    Full-block mathematics is evaluated with exact rational operands and then
    rounded once to FP64. It is independent of the two-thread split, F32 FMA
    and wave reduction used in lane_model.
    """
    total = Fraction(0)
    for d4, d8, s8, ns, qs in decode(q4, q8):
        dot = sum(n*q for n, q in zip(ns, qs))
        total += Fraction(d4) * (Fraction(d8)*dot - 8*Fraction(s8))
    return finite(float(total))


def lane_model(q4, q8):
    blocks = decode(q4, q8)
    waves = [[0.0] * 32 for _ in range(4)]
    for wave in range(4):
        for lane in range(32):
            accumulator = 0.0
            for block in range((32*wave+lane)//2, len(blocks), 64):
                d4, d8, s8, ns, qs = blocks[block]
                first = 8*(lane % 2)
                dot = sum(ns[j]*qs[j] + ns[j+16]*qs[j+16] for j in range(first, first+8))
                partial = fma(float(dot), d8, -4.0*s8)
                accumulator = fma(partial, d4, accumulator)
            waves[wave][lane] = accumulator
    combined = [add(add(add(waves[0][i], waves[1][i]), waves[2][i]), waves[3][i]) for i in range(32)]
    return butterfly(combined)


def error_budget(q4, q8):
    """A priori normwise bound, not fitted tolerance; no relative-error division.

    For each path: one partial FMA + at most ceil(blocks/64) accumulator
    roundings + three LDS adds + five butterfly adds = iterations+9.
    Use gamma_(iterations+12) to also cover final FP64 oracle rounding and
    bound conversion. Magnitude includes both operands BEFORE cancellation.
    Integer dot and -4*half-s are exact. Inputs/intermediates outside normal
    F32/finite half scope (including half subnormals) are rejected. No underflow floor needed.
    """
    blocks = decode(q4, q8)
    magnitude = Fraction(0)
    for d4, d8, s8, ns, qs in blocks:
        for first in (0, 8):
            dot = sum(ns[j]*qs[j] + ns[j+16]*qs[j+16] for j in range(first, first+8))
            magnitude += abs(Fraction(d4)) * (abs(Fraction(d8)*dot) + 4*abs(Fraction(s8)))
    n = (len(blocks)+63)//64 + 12
    if n*U >= 1:
        raise ValueError('rounding bound outside supported range')
    bound = n*U/(1-n*U)*magnitude
    return math.nextafter(float(bound), math.inf) if bound else 0.0


def verify_output(value, q4, q8):
    value = finite(value)
    modeled = lane_model(q4, q8)  # also validates supported intermediate range
    oracle = semantic_fp64(q4, q8)
    budget = error_budget(q4, q8)
    if abs(modeled-oracle) > budget:
        raise ValueError('lane reference violates predeclared bound')
    if abs(value-oracle) > budget:
        raise ValueError('output outside predeclared normwise bound')
    return dict(fp64=oracle, lane_f32=modeled, absolute_error=abs(value-oracle), budget=budget)


PROFILE = dict(schema=2, widths=list(WIDTHS), q8_block_bytes=36, q4_block_bytes=18,
               q8_rounding="F32 RN-even division; integer ties away; original XOR16,8,4,2,1 sum",
               q4_rounding="two F32 fused operations per partial; stride64; waves1,2,3; XOR16,8,4,2,1",
               budget="gamma_(ceil(blocks/64)+12) * sum_abs_uncancelled_partial_operands",
               unsupported=["nonfinite", "F32 subnormal intermediates", "binary16 underflow-to-zero/overflow", "fused pointer arguments"],
               gpu_executed=False)


def quantize_q8_1(input_f32_bytes):
    if not input_f32_bytes or len(input_f32_bytes) % 128:
        raise ValueError('complete32-value F32 blocks required')
    return quantize_q8(struct.unpack('<' + 'f'*(len(input_f32_bytes)//4), input_f32_bytes))


def reference_q4(q4bytes, q8bytes, width, rows):
    if width not in WIDTHS or type(rows) is not int or rows <= 0:
        raise ValueError('unsupported reference dimensions')
    row_bytes = width//32*18
    if len(q4bytes) != rows*row_bytes or len(q8bytes) != width//32*36:
        raise ValueError('reference dimensions/bytes differ')
    output = dict(fp64=[], lane_f32=[], budget=[])
    for row in range(rows):
        weights = q4bytes[row*row_bytes:(row+1)*row_bytes]
        values = verify_output(lane_model(weights, q8bytes), weights, q8bytes)
        for key in output:
            output[key].append(values[key])
    return output
