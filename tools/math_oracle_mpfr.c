/* Offline reference generator. No HIP, target code or host libm approximation. */
#include <float.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpfr.h>

static void require(int condition, const char *message) {
    if (!condition) { fprintf(stderr, "%s\n", message); exit(1); }
}
static uint32_t bits(float value) { uint32_t out; memcpy(&out, &value, 4); return out; }
static void decode(mpfr_t out, uint32_t word) {
    unsigned exponent = (word >> 23) & 255, fraction = word & 0x7fffff;
    if (exponent == 255) {
        if (fraction) mpfr_set_nan(out); else mpfr_set_inf(out, word >> 31 ? -1 : 1);
    } else if (!exponent && !fraction) {
        mpfr_set_zero(out, word >> 31 ? -1 : 1);
    } else {
        require(mpfr_set_ui_2exp(out, fraction | (exponent ? 0x800000 : 0),
            exponent ? (long)exponent - 150 : -149, MPFR_RNDN) == 0, "input not exact");
        if (word >> 31) mpfr_neg(out, out, MPFR_RNDN);
    }
}
static void emit(unsigned index, unsigned op, unsigned rule, uint32_t lo, uint32_t hi,
                 uint32_t reference, unsigned precision, long ulp, const char *left, const char *right) {
    printf("{\"index\":%u,\"operation\":%u,\"rule\":%u,\"lo\":%" PRIu32
           ",\"hi\":%" PRIu32 ",\"reference\":%" PRIu32 ",\"precision\":%u,"
           "\"ulp_exponent\":%ld,\"reference_lo\":\"%s\",\"reference_hi\":\"%s\"}\n",
           index, op, rule, lo, hi, reference, precision, ulp, left, right);
}
static void exact(unsigned index, unsigned op, uint32_t value) {
    emit(index, op, 0, value, value, value, 0, 0, "analytic", "analytic");
}
static void nan_result(unsigned index, unsigned op) {
    emit(index, op, 1, 0, 0, 0x7fc00000, 0, 0, "quiet_nan", "quiet_nan");
}
static int evaluate(mpfr_t out, unsigned op, mpfr_t x, mpfr_t argument,
                    mpfr_t factor, mpfr_t offset, mpfr_rnd_t rounding) {
    switch (op) {
    case 0: return mpfr_fma(out, x, factor, offset, rounding);
    case 1: return mpfr_sqrt(out, argument, rounding);
    case 2: return mpfr_sin(out, x, rounding);
    case 3: return mpfr_exp(out, x, rounding);
    case 4: return mpfr_log(out, argument, rounding);
    default: return mpfr_tanh(out, x, rounding);
    }
}
static void one(unsigned index, unsigned op, uint32_t input) {
    const uint32_t magnitude = input & 0x7fffffff, sign = input & 0x80000000;
    if (magnitude > 0x7f800000) { nan_result(index, op); return; }
    if (magnitude == 0x7f800000) {
        if (op == 2) nan_result(index, op);
        else exact(index, op, op == 0 ? input : op == 5 ? sign | 0x3f800000 :
                   op == 3 && sign ? 0 : 0x7f800000);
        return;
    }
    if (!magnitude && (op == 2 || op == 5 || op == 3)) {
        exact(index, op, op == 3 ? 0x3f800000 : input); return;
    }
    /* |sin(x)-x| and |tanh(x)-x| < |x|^3 here, far below half a binary32 ulp. */
    if (magnitude <= 0x00800000 && (op == 2 || op == 5)) {
        exact(index, op, input); return;
    }
    /* These thresholds are independently proved at startup, not MPFR range errors. */
    if (op == 3 && ((!sign && magnitude >= 0x43000000) || (sign && magnitude >= 0x43800000))) {
        exact(index, op, sign ? 0 : 0x7f800000); return;
    }
    for (unsigned precision = 256; precision <= 2048; precision *= 2) {
        mpfr_t x, argument, factor, offset, lower, upper, temp, tolerance;
        mpfr_inits2(precision, x, argument, factor, offset, lower, upper, temp, tolerance, (mpfr_ptr)0);
        decode(x, input);
        mpfr_abs(argument, x, MPFR_RNDN);
        require(mpfr_add_d(argument, argument, 0.25, MPFR_RNDN) == 0, "pre-add not exact at working precision");
        decode(argument, bits(mpfr_get_flt(argument, MPFR_RNDN))); /* Required binary32 pre-round. */
        mpfr_set_d(factor, 1.25, MPFR_RNDN); mpfr_set_d(offset, -0.5, MPFR_RNDN);
        mpfr_clear_flags();
        evaluate(lower, op, x, argument, factor, offset, MPFR_RNDD);
        evaluate(upper, op, x, argument, factor, offset, MPFR_RNDU);
        require(!mpfr_overflow_p() && !mpfr_underflow_p() && mpfr_number_p(lower) && mpfr_number_p(upper),
                "MPFR range/invalid result must not masquerade as binary32 classification");
        uint32_t reference = bits(mpfr_get_flt(lower, MPFR_RNDN));
        int resolved = reference == bits(mpfr_get_flt(upper, MPFR_RNDN));
        unsigned rule = 0; uint32_t low_bits = reference, high_bits = reference; long ulp = 0;
        if (resolved && op >= 2 && (reference & 0x7fffffff) && (reference & 0x7fffffff) < 0x7f800000 &&
            !(op == 5 && magnitude >= 0x41800000)) {
            mpfr_abs(temp, lower, MPFR_RNDN); long e1 = mpfr_get_exp(temp);
            mpfr_abs(temp, upper, MPFR_RNDN); long e2 = mpfr_get_exp(temp);
            resolved = e1 == e2 && mpfr_sgn(lower) == mpfr_sgn(upper);
            if (resolved) {
                ulp = e1 - 24; if (ulp < -149) ulp = -149;
                unsigned budget = op == 2 ? 4 : op == 5 ? 5 : 3;
                mpfr_set_ui_2exp(tolerance, budget, ulp, MPFR_RNDN);
                /* Inner envelope certifies every included float for all y in [lower,upper]. */
                mpfr_sub(temp, upper, tolerance, MPFR_RNDU);
                low_bits = bits(mpfr_get_flt(temp, MPFR_RNDU));
                mpfr_add(temp, lower, tolerance, MPFR_RNDD);
                high_bits = bits(mpfr_get_flt(temp, MPFR_RNDD));
                if (low_bits == 0xff800000) low_bits = 0xff7fffff;
                if (high_bits == 0x7f800000) high_bits = 0x7f7fffff;
                /* The ULP envelope never licenses the wrong sign or flushing a nonzero result. */
                if (!(reference & 0x80000000) && ((low_bits & 0x80000000) || !low_bits)) low_bits = 1;
                if ((reference & 0x80000000) && (!(high_bits & 0x80000000) || high_bits == 0x80000000))
                    high_bits = 0x80000001;
                rule = 2;
            }
        }
        if (resolved) {
            char left[640], right[640];
            require(mpfr_snprintf(left, sizeof(left), "%Ra", lower) > 0, "format failure");
            require(mpfr_snprintf(right, sizeof(right), "%Ra", upper) > 0, "format failure");
            emit(index, op, rule, low_bits, high_bits, reference, precision, ulp, left, right);
        }
        mpfr_clears(x, argument, factor, offset, lower, upper, temp, tolerance, (mpfr_ptr)0);
        if (resolved) return;
    }
    require(0, "reference classification, binade or rounding unresolved at2048bits");
}
int main(int argc, char **argv) {
    require(argc == 2 && sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
            "expected input file and binary32 host float");
    require(strcmp(mpfr_get_version(), MPFR_VERSION_STRING) == 0 && strcmp(MPFR_VERSION_STRING, "4.2.2") == 0,
            "MPFR header/runtime version mismatch");
    fprintf(stderr, "MPFR=%s GMP=%s\n", mpfr_get_version(), gmp_version);
    mpfr_t x, y, bound; mpfr_inits2(256, x, y, bound, (mpfr_ptr)0);
    mpfr_set_ui(x, 128, MPFR_RNDN); mpfr_exp(y, x, MPFR_RNDD); mpfr_set_ui_2exp(bound, 1, 128, MPFR_RNDN);
    require(mpfr_cmp(y, bound) > 0, "exp overflow threshold proof failed");
    mpfr_set_si(x, -256, MPFR_RNDN); mpfr_exp(y, x, MPFR_RNDU); mpfr_set_ui_2exp(bound, 1, -150, MPFR_RNDN);
    require(mpfr_cmp(y, bound) < 0, "exp underflow threshold proof failed");
    mpfr_clears(x, y, bound, (mpfr_ptr)0);
    FILE *file = fopen(argv[1], "r"); require(file != NULL, "input open failed");
    uint32_t value; unsigned index = 0;
    while (fscanf(file, "%" SCNx32, &value) == 1) {
        require(index < 257, "too many inputs");
        for (unsigned op = 0; op < 6; ++op) one(index, op, value);
        ++index;
    }
    require(index == 257 && feof(file), "wrong input count or malformed token");
    require(fclose(file) == 0, "input close failed");
    return 0;
}
