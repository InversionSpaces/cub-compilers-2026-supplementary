#!/usr/bin/env python3
"""
Fuzz tester for the Lama compiler.

Generates random arithmetic expressions over random variables, compiles and
runs them through the interpreter (-i), stack machine (-s), and x86_64
native code, then verifies all three outputs match the expected result
computed in Python.

Usage:
    python3 fuzz_test.py -n 3 -d 3 -k 10 [--seed 42]

Options:
    -n N              Max number of variables (default: 3)
    -d D              Max expression depth (default: 3)
    -k K              Number of random test iterations (default: 10)
    --seed S          Random seed for reproducibility (optional)
    --input-range R   Max absolute value of random inputs (default: 100)
    --const-range R   Max absolute value of random constants (default: 10)
    --ops GROUPS      Comma-separated operator groups to use (default: arith,cmp,logic)
                      Available groups: arith (+,-,*), div (/,%),
                      cmp (==,!=,<,<=,>,>=), logic (&&,!!), all

On mismatch the script exits with code 1 and preserves the generated
.lama and .input files for inspection.  On success all temporary files
are deleted after each iteration.
"""

import argparse
import math
import os
import random
import subprocess
import sys

VERBOSE = False


def log(msg):
    """Print a debug message when verbose mode is enabled."""
    if VERBOSE:
        print(f"  [LOG] {msg}", flush=True)


def log_subprocess(label, ret):
    """Log command, return code, stdout, and stderr of a subprocess result."""
    if not VERBOSE:
        return
    log(f"{label}: exit code {ret.returncode}")
    if ret.stdout and ret.stdout.strip():
        for line in ret.stdout.strip().splitlines():
            log(f"  stdout: {line}")
    if ret.stderr and ret.stderr.strip():
        for line in ret.stderr.strip().splitlines():
            log(f"  stderr: {line}")

# ---------------------------------------------------------------------------
# Expression tree helpers
# ---------------------------------------------------------------------------

# Arithmetic operators that are always safe (no division-by-zero)
SAFE_OPS = ["+", "-", "*"]
# Operators that can cause division by zero
UNSAFE_OPS = ["/", "%"]
# Comparison operators (return 0 or 1)
CMP_OPS = ["==", "!=", "<", "<=", ">", ">="]
# Logical operators (return 0 or 1)
LOGIC_OPS = ["&&", "!!"]

ALL_OPS = SAFE_OPS + UNSAFE_OPS + CMP_OPS + LOGIC_OPS

# Default weights per operator.  Arithmetic ops get higher weight so that
# expressions don't collapse to 0/1 too quickly via cascading comparisons
# or logical operators.
OP_WEIGHTS = {
    "+": 5, "-": 5, "*": 5,
    "/": 2, "%": 2,
    "==": 1, "!=": 1, "<": 1, "<=": 1, ">": 1, ">=": 1,
    "&&": 1, "!!": 1,
}


def generate_var_names(n):
    """Return a list of *n* distinct variable names."""
    names = []
    for i in range(n):
        if i < 26:
            names.append(chr(ord("a") + i))
        else:
            names.append(f"v{i}")
    return names


OP_GROUPS = {
    "arith": SAFE_OPS,
    "div":   UNSAFE_OPS,
    "cmp":   CMP_OPS,
    "logic": LOGIC_OPS,
    "all":   ALL_OPS,
}


def resolve_ops(spec):
    """
    Parse an operator-group specification string.

    *spec* is a comma-separated list of group names (arith, div, cmp, logic)
    or the keyword ``all``.  Returns the flat list of operator strings and
    their corresponding weights for weighted random selection.
    """
    ops = []
    for name in spec.split(","):
        name = name.strip()
        if name not in OP_GROUPS:
            raise ValueError(
                f"Unknown op group {name!r}; choose from {list(OP_GROUPS)}"
            )
        ops.extend(OP_GROUPS[name])
    weights = [OP_WEIGHTS[op] for op in ops]
    return ops, weights


def generate_expr(variables, max_depth, const_range, ops, weights, *, _depth=0):
    """
    Build a random expression tree represented as nested tuples.

    Leaf  = ("var", name)  |  ("const", int)
    Node  = ("binop", op_str, left, right)

    *ops* and *weights* are parallel lists; operators are chosen via
    weighted random selection so that arithmetic ops dominate over
    comparison/logic ops (which collapse values to 0/1).
    """
    # Decide whether to generate a leaf.
    # Probability of stopping grows linearly with depth: 0 at root, 1 at max_depth.
    leaf_prob = _depth / max_depth if max_depth > 0 else 1.0
    if _depth >= max_depth or (_depth > 0 and random.random() < leaf_prob):
        if variables and random.random() < 0.6:
            return ("var", random.choice(variables))
        else:
            return ("const", random.randint(-const_range, const_range))

    op = random.choices(ops, weights=weights, k=1)[0]
    left = generate_expr(variables, max_depth, const_range, ops, weights, _depth=_depth + 1)
    right = generate_expr(variables, max_depth, const_range, ops, weights, _depth=_depth + 1)
    return ("binop", op, left, right)


# ---------------------------------------------------------------------------
# Conversion to Lama source
# ---------------------------------------------------------------------------

def expr_to_lama(expr):
    """Convert an expression tree to Lama syntax (for embedding)."""
    tag = expr[0]
    if tag == "var":
        return f'"{expr[1]}"'
    elif tag == "const":
        val = expr[1]
        # Negative constants must be written as (0 - N) because the
        # Embedding.meta redefines `-` as an infix operator, and the
        # Lama parser may mis-handle a bare negative literal in that
        # context.  Existing regression tests follow the same pattern.
        if val < 0:
            return f"(0 - {-val})"
        return str(val)
    else:  # binop
        op, left, right = expr[1], expr[2], expr[3]
        return f"({expr_to_lama(left)} {op} {expr_to_lama(right)})"


def build_lama_program(variables, expr):
    """
    Return the full Lama program text (to be placed in a .lama file).

    The programme reads every variable, assigns the expression result to
    ``result``, and writes it.
    """
    parts = []
    for v in variables:
        parts.append(f'read ("{v}")')
    parts.append(f'"result" ::= {expr_to_lama(expr)}')
    parts.append('write ("result")')
    return " >>\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Python-side expression evaluation (must match Lama semantics)
# ---------------------------------------------------------------------------

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_I64_MOD = 1 << 64


def _wrap64(n):
    """Wrap an arbitrary Python int to a signed 64-bit value (Lama semantics)."""
    n = n % _I64_MOD          # bring into 0 .. 2^64-1
    if n > _I64_MAX:
        n -= _I64_MOD
    return n


def c_style_div(a, b):
    """Integer division that truncates toward zero (C / Lama semantics)."""
    if b == 0:
        raise ZeroDivisionError
    return int(math.copysign(abs(a) // abs(b), a * b)) if (a * b) else 0


def c_style_mod(a, b):
    """Modulo with sign-of-dividend (C / Lama semantics)."""
    if b == 0:
        raise ZeroDivisionError
    return a - c_style_div(a, b) * b


def eval_expr(expr, env):
    """Evaluate *expr* with variable bindings *env*; returns a Python int
    wrapped to 64-bit signed range to match Lama."""
    tag = expr[0]
    if tag == "var":
        return env[expr[1]]
    elif tag == "const":
        return expr[1]
    else:  # binop
        op = expr[1]
        lv = eval_expr(expr[2], env)
        rv = eval_expr(expr[3], env)
        if op == "+":
            return _wrap64(lv + rv)
        elif op == "-":
            return _wrap64(lv - rv)
        elif op == "*":
            return _wrap64(lv * rv)
        elif op == "/":
            return _wrap64(c_style_div(lv, rv))
        elif op == "%":
            return _wrap64(c_style_mod(lv, rv))
        elif op == "==":
            return 1 if lv == rv else 0
        elif op == "!=":
            return 1 if lv != rv else 0
        elif op == "<":
            return 1 if lv < rv else 0
        elif op == "<=":
            return 1 if lv <= rv else 0
        elif op == ">":
            return 1 if lv > rv else 0
        elif op == ">=":
            return 1 if lv >= rv else 0
        elif op == "&&":
            return 1 if (lv != 0 and rv != 0) else 0
        elif op == "!!":
            return 1 if (lv != 0 or rv != 0) else 0
        else:
            raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# Running a Lama test through the build system
# ---------------------------------------------------------------------------

REGRESSION_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_OBJ = os.path.join(REGRESSION_DIR, "..", "runtime64", "runtime.o")
SRC_DIR = os.path.join(REGRESSION_DIR, "..", "src")
EMBEDDING = os.path.join(REGRESSION_DIR, "Embedding.meta")
TEST_NAME = "fuzz_tmp"


def write_test_files(lama_source, input_values):
    """Write the .lama and .input files into the regression directory."""
    lama_path = os.path.join(REGRESSION_DIR, f"{TEST_NAME}.lama")
    input_path = os.path.join(REGRESSION_DIR, f"{TEST_NAME}.input")
    with open(lama_path, "w") as f:
        f.write(lama_source)
    with open(input_path, "w") as f:
        for v in input_values:
            f.write(f"{v}\n")


def compile_test():
    """
    Compile the fuzz test using the same pipeline as the Makefile.

    Returns True on success, prints errors and returns False on failure.
    """
    lama_path = os.path.join(REGRESSION_DIR, f"{TEST_NAME}.lama")

    # Step 1: cpp preprocessing (embed into Embedding.meta)
    # We use bash explicitly for command substitution
    cpp_cmd = (
        f'cpp -P -D PROGRAM_BODY="$(tr -d \'\\n\' < {lama_path})" '
        f"{EMBEDDING} > {os.path.join(REGRESSION_DIR, 'tmp.lama')}"
    )
    log(f"cpp cmd: {cpp_cmd}")
    ret = subprocess.run(["bash", "-c", cpp_cmd], capture_output=True, text=True)
    log_subprocess("cpp", ret)
    if ret.returncode != 0:
        print(f"  [FAIL] cpp preprocessing failed:\n{ret.stderr}", file=sys.stderr)
        return False

    # Step 2: lamac compilation
    tmp_lama = os.path.join(REGRESSION_DIR, "tmp.lama")
    binary = os.path.join(REGRESSION_DIR, TEST_NAME)
    lamac_cmd = ["lamac", "-I", SRC_DIR, "-o", binary, tmp_lama, "-ds"]
    log(f"lamac cmd: {' '.join(lamac_cmd)}")
    ret = subprocess.run(lamac_cmd, capture_output=True, text=True)
    log_subprocess("lamac", ret)
    os.remove(tmp_lama)
    if ret.returncode != 0:
        print(f"  [FAIL] lamac compilation failed:\n{ret.stderr}", file=sys.stderr)
        return False

    return True


def run_mode_interp(input_data):
    """Run with -i (interpreter) and return stdout."""
    binary = os.path.join(REGRESSION_DIR, TEST_NAME)
    log(f"interpreter cmd: {binary} -i")
    ret = subprocess.run(
        [binary, "-i"], input=input_data, capture_output=True, text=True,
    )
    log_subprocess("interpreter", ret)
    if ret.returncode != 0:
        return None, ret.stderr
    return ret.stdout.strip(), None


def run_mode_sm(input_data):
    """Run with -s (stack machine) and return stdout."""
    binary = os.path.join(REGRESSION_DIR, TEST_NAME)
    log(f"stack machine cmd: {binary} -s")
    ret = subprocess.run(
        [binary, "-s"], input=input_data, capture_output=True, text=True,
    )
    log_subprocess("stack machine", ret)
    if ret.returncode != 0:
        return None, ret.stderr
    return ret.stdout.strip(), None


def run_mode_x86(input_data):
    """Compile to x86_64 assembly, link, run, and return stdout."""
    binary = os.path.join(REGRESSION_DIR, TEST_NAME)
    asm_file = os.path.join(REGRESSION_DIR, f"{TEST_NAME}.s")
    run_file = os.path.join(REGRESSION_DIR, f"{TEST_NAME}.run")

    # Generate assembly
    log(f"asm generation cmd: {binary}")
    ret = subprocess.run([binary], capture_output=True, text=True)
    log_subprocess("asm generation", ret)
    if ret.returncode != 0:
        return None, f"asm generation failed: {ret.stderr}"
    with open(asm_file, "w") as f:
        f.write(ret.stdout)

    # Link with gcc
    gcc_cmd = ["gcc", "-g", "-o", run_file, RUNTIME_OBJ, asm_file,
               "-z", "noexecstack"]
    log(f"gcc cmd: {' '.join(gcc_cmd)}")
    ret = subprocess.run(gcc_cmd, capture_output=True, text=True)
    log_subprocess("gcc", ret)
    if ret.returncode != 0:
        return None, f"gcc linking failed: {ret.stderr}"

    # Run
    log(f"native run cmd: {run_file}")
    ret = subprocess.run(
        [run_file], input=input_data, capture_output=True, text=True,
    )
    log_subprocess("native run", ret)
    if ret.returncode != 0:
        return None, f"native execution failed: {ret.stderr}"
    return ret.stdout.strip(), None


def cleanup_test_files():
    """Remove all generated files for the fuzz test."""
    extensions = [".lama", ".input", ".log", ".s", ".run", ".i", ".sm", ""]
    for ext in extensions:
        path = os.path.join(REGRESSION_DIR, f"{TEST_NAME}{ext}")
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fuzz tester for the Lama compiler"
    )
    parser.add_argument("-n", type=int, default=3,
                        help="Max number of variables (default: 3)")
    parser.add_argument("-d", type=int, default=3,
                        help="Max expression depth (default: 3)")
    parser.add_argument("-k", type=int, default=10,
                        help="Number of test iterations (default: 10)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--input-range", type=int, default=10000,
                        help="Max absolute input value (default: 100)")
    parser.add_argument("--const-range", type=int, default=10000,
                        help="Max absolute constant value (default: 10)")
    parser.add_argument("--ops", type=str, default="arith,cmp,logic",
                        help="Comma-separated op groups: arith,div,cmp,logic,all "
                             "(default: arith,cmp,logic)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging of subprocesses")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    ops, weights = resolve_ops(args.ops)

    if args.seed is not None:
        random.seed(args.seed)

    passed = 0
    for iteration in range(1, args.k + 1):
        # 1. Pick a random number of variables (1..n)
        num_vars = random.randint(1, args.n)
        variables = generate_var_names(num_vars)

        # 2. Generate a random expression
        expr = generate_expr(variables, args.d, args.const_range, ops, weights)

        # 3. Generate random input values
        env = {v: random.randint(-args.input_range, args.input_range)
               for v in variables}
        input_values = [env[v] for v in variables]
        input_data = "\n".join(str(v) for v in input_values) + "\n"

        # 4. Evaluate in Python
        try:
            expected = eval_expr(expr, env)
        except ZeroDivisionError:
            # Skip this iteration — unlucky constant/variable combo
            print(f"[{iteration}/{args.k}] skipped (division by zero in Python eval)")
            continue

        # 5. Build the Lama source
        lama_source = build_lama_program(variables, expr)

        print(f"[{iteration}/{args.k}] Testing: vars={env}  expr={expr_to_lama(expr)}  expected={expected}")

        # 6. Write files and compile
        write_test_files(lama_source, input_values)
        log(f"Wrote {TEST_NAME}.lama and {TEST_NAME}.input")
        if not compile_test():
            print(f"\n=== COMPILATION FAILURE on iteration {iteration} ===")
            print(f"  Expression (Lama): {expr_to_lama(expr)}")
            print(f"  Variables:         {env}")
            print(f"  Lama source kept in {TEST_NAME}.lama")
            sys.exit(1)

        # 7. Run in all three modes and check
        modes = [
            ("interpreter (-i)", run_mode_interp),
            ("stack machine (-s)", run_mode_sm),
            ("x86_64 native", run_mode_x86),
        ]

        failed = False
        for mode_name, run_fn in modes:
            log(f"Running mode: {mode_name}")
            result, err = run_fn(input_data)
            if err is not None:
                print(f"\n=== RUNTIME ERROR on iteration {iteration} [{mode_name}] ===")
                print(f"  Error:      {err}")
                print(f"  Expression: {expr_to_lama(expr)}")
                print(f"  Variables:  {env}")
                print(f"  Expected:   {expected}")
                print(f"  Lama source kept in {TEST_NAME}.lama / {TEST_NAME}.input")
                sys.exit(1)

            # The output may contain multiple lines; we expect exactly one
            try:
                actual = int(result)
            except (ValueError, TypeError):
                print(f"\n=== BAD OUTPUT on iteration {iteration} [{mode_name}] ===")
                print(f"  Raw output: {result!r}")
                print(f"  Expression: {expr_to_lama(expr)}")
                print(f"  Variables:  {env}")
                print(f"  Expected:   {expected}")
                print(f"  Lama source kept in {TEST_NAME}.lama / {TEST_NAME}.input")
                sys.exit(1)

            if actual != expected:
                print(f"\n=== MISMATCH on iteration {iteration} [{mode_name}] ===")
                print(f"  Expression: {expr_to_lama(expr)}")
                print(f"  Variables:  {env}")
                print(f"  Expected (Python): {expected}")
                print(f"  Actual   (Lama):   {actual}")
                print(f"  Lama source kept in {TEST_NAME}.lama / {TEST_NAME}.input")
                failed = True
                break

        if failed:
            sys.exit(1)

        # 8. All modes agree — clean up and continue
        cleanup_test_files()
        passed += 1
        print(f"[{iteration}/{args.k}] PASS  result={expected}")

    print(f"\nAll {passed} tests passed.")


if __name__ == "__main__":
    main()
