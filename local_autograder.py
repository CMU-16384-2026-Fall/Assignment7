"""Autograder for HW7: Denavit-Hartenberg forward kinematics of the xArm7 (ex1).

    python local_autograder.py             # grade everything
    python local_autograder.py -p ex1      # one exercise
    python local_autograder.py -p vis1     # run ex1 on the simulated xArm7

The graded modes need only numpy. The vis mode needs the course's robot
library, installed from the copy in this handout:
    cd 16384-robot-lib && conda env update -f environment.yml && cd ..
    conda activate 16384
"""

import argparse
import contextlib
import importlib.util
import inspect
import json
import signal
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
# Forward kinematics is closed form, so a correct answer matches to rounding.
TOL = 1e-6

# Student code that hangs fails the case rather than the run. One case that
# runs this long fails, and so do the rest of that test, untried: the same
# code would only hang again, once per case.
CASE_TIME_LIMIT = 10  # s
IMPORT_TIME_LIMIT = 30  # s

MAX_FAILURES_SHOWN = 10
# Long enough for a failure to say where in the chain it went wrong, and why.
MAX_DETAIL_CHARS = 600


class StudentTimeout(Exception):
    """Student code ran past its time limit."""


@contextlib.contextmanager
def _time_limit(seconds, what):
    """Raise StudentTimeout if the block runs longer than `seconds`.

    A SIGALRM timer, so only where there is one: Unix, on the main thread.
    Elsewhere the block simply runs.
    """
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        yield
        return

    def expired(signum, frame):
        raise StudentTimeout(f"{what} ran for more than {seconds} s (an infinite loop?)")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _load_module(path):
    """Import a standalone script by path, with its own folder first on sys.path."""
    folder = str(Path(path).resolve().parent)
    sys.modules.pop("robot", None)
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location(Path(path).stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(folder)
        sys.modules.pop("robot", None)
    return module


def _read_jsonl(path):
    """Stream one JSON object per line."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _close(actual, expected, tol=TOL):
    """np.allclose that fails (rather than raising) on garbage/missing values.

    Shapes must match exactly: allclose alone would broadcast a scalar or a
    (4, 1) against a (4, 4) and call it a match.
    """
    try:
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        return actual.shape == expected.shape and bool(
            np.allclose(actual, expected, atol=tol, rtol=0)
        )
    except (TypeError, ValueError):
        return False


def _array(value, name, shape):
    """`value` as a float array of `shape`, or a ValueError saying what it was."""
    if value is None:
        raise ValueError(f"{name} is None: did the function reach its return statement?")
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    return array


def _sentence(text):
    """`text` with its first letter capitalised, to follow a full stop."""
    return text[:1].upper() + text[1:] if text else text


def _call(name, args):
    return f"{name}(" + ", ".join(f"{v:.4g}" for v in args) + ")"


def _right_before(i, what):
    """'frame 0 is ...' or 'frames 0-3 are ...'."""
    return f"frame 0 {what[0]}" if i == 1 else f"frames 0-{i - 1} {what[1]}"


def _describe(err):
    if isinstance(err, SystemExit):
        return f"your code called sys.exit() or exit() (code {err.code!r}); it must return instead"
    return f"{type(err).__name__}: {err}"


def _fmt(values):
    return "[" + ", ".join(f"{v:.4g}" for v in np.ravel(values)) + "]"


def _transform_problem(H, want, tol):
    """What is wrong with the 4x4 `H` compared with `want`, or None."""
    if not np.all(np.isfinite(H)):
        return f"it has NaN or inf in it: {_fmt(H[~np.isfinite(H)][:4])}"
    if not _close(H[3], [0, 0, 0, 1], tol):
        return f"the bottom row should be [0, 0, 0, 1], got {_fmt(H[3])}"
    if not _close(H[:3, :3], want[:3, :3], tol):
        return (f"the rotation block is off by up to "
                f"{np.max(np.abs(H[:3, :3] - want[:3, :3])):.3g}")
    if not _close(H[:3, 3], want[:3, 3], tol):
        return (f"the position is {_fmt(H[:3, 3])}, expected {_fmt(want[:3, 3])} "
                f"({np.linalg.norm(H[:3, 3] - want[:3, 3]) * 1000:.1f} mm off)")
    return None


class Grader:
    """Counts cases and keeps only the failures it is going to print."""

    def __init__(self, name, keep=MAX_FAILURES_SHOWN):
        self.name = name
        self.passed = 0
        self.total = 0
        self.keep = keep
        self.failures = []  # (label, detail), capped at `keep`

    def record(self, label, passed, detail=""):
        self.total += 1
        if passed:
            self.passed += 1
        elif len(self.failures) < self.keep:
            self.failures.append((label, detail))

    def run(self, cases, label, check):
        """Record `check(case, expected)` -> (ok, detail) for every case.

        Anything the student's code raises is a failure of that case, not of
        the run.
        """
        cases = iter(cases)
        for i, (case, expected) in enumerate(cases):
            try:
                with _time_limit(CASE_TIME_LIMIT, "your code, on this case,"):
                    ok, detail = check(case, expected)
            except StudentTimeout as err:
                self.record(f"{label(i, case)} and every case after it", False,
                            f"{err}: the rest of this test is failed without being run")
                self.total += sum(1 for _ in cases)
                return self
            except KeyboardInterrupt:
                raise
            except BaseException as err:  # SystemExit too: student code can't end the run
                ok, detail = False, _describe(err)
            self.record(label(i, case), ok, "" if ok else detail)
        return self

    def report(self, max_failures=MAX_FAILURES_SHOWN):
        """Print the result: a passing run is one line, a failing one lists the
        failures it kept and says how many it didn't."""
        print(f"\n=== {self.name}: {self.passed}/{self.total} ===")

        shown = min(max_failures, len(self.failures))
        for label, detail in self.failures[:shown]:
            line = f"  [FAIL] {label}"
            if detail:
                if len(detail) > MAX_DETAIL_CHARS:
                    detail = detail[:MAX_DETAIL_CHARS] + "..."
                line += f" — {detail}"
            print(line)

        hidden = (self.total - self.passed) - shown
        if hidden > 0:
            print(f"  ... and {hidden} more failure(s); pass --verbose to keep more")
        return self.passed, self.total


# ----------------------------------------------------------------------
# The tests. Each takes the student's Robot class and an iterable of
# (case, expected) pairs, so Gradescope's grade.py runs exactly these on its
# perturbed cases.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Part (a): the DH transform
# ----------------------------------------------------------------------

# The four elementary transforms H_i^{i-1} is the product of, exactly as the
# writeup's Background section prints them. A failing dh_transform is tried on
# one parameter at a time, against these, to find which factor is wrong.


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _trans_z(d):
    H = np.eye(4)
    H[2, 3] = d
    return H


def _trans_x(a):
    H = np.eye(4)
    H[0, 3] = a
    return H


def _rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def _dh_function(robot_cls):
    """Robot.dh_transform as a plain function of (a, alpha, d, theta).

    The stub declares it a @staticmethod; if a student deleted the decorator
    and gave it a `self`, it is called through an instance instead, so the
    grade is about the transform, not the decorator.
    """
    raw = inspect.getattr_static(robot_cls, "dh_transform", None)
    if raw is None:
        raise AttributeError("Robot has no dh_transform method")
    if isinstance(raw, (staticmethod, classmethod)):
        return robot_cls.dh_transform
    return robot_cls(np.zeros((1, 4))).dh_transform


def _call_dh(dh_fn, a, alpha, d, theta):
    return _array(dh_fn(a, alpha, d, theta), "dh_transform's result", (4, 4))


def _diagnose_dh(dh_fn, a, alpha, d, theta):
    """Which factor of Rot_z(theta) Trans_z(d) Trans_x(a) Rot_x(alpha) is wrong."""
    pieces = [
        ("Rot_z(theta), the first factor", (0.0, 0.0, 0.0, theta), _rot_z(theta),
         "a rotation by theta about z"),
        ("Trans_z(d), the second factor", (0.0, 0.0, d, 0.0), _trans_z(d),
         "a translation by d along z"),
        ("Trans_x(a), the third factor", (a, 0.0, 0.0, 0.0), _trans_x(a),
         "a translation by a along x"),
        ("Rot_x(alpha), the fourth factor", (0.0, alpha, 0.0, 0.0), _rot_x(alpha),
         "a rotation by alpha about x"),
    ]
    for name, args, want, meaning in pieces:
        try:
            got = _call_dh(dh_fn, *args)
        except Exception as err:
            return f"{name}: {_call('dh_transform', args)} raised {_describe(err)}"
        problem = _transform_problem(got, want, TOL)
        if problem is not None:
            return (f"{name}, is wrong. With every other parameter 0, "
                    f"{_call('dh_transform', args)} should be just {meaning}, but {problem}")
    return ("none on its own: each factor is right with the other parameters 0, so they "
            "are multiplied in the wrong order. H = Rot_z(theta) Trans_z(d) Trans_x(a) "
            "Rot_x(alpha), in that order")


def grade_dh_transform(robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        args = (case["a"], case["alpha"], case["d"], case["theta"])
        dh_fn = _dh_function(robot_cls)
        H = _call_dh(dh_fn, *args)
        problem = _transform_problem(H, np.asarray(expected["H"]), TOL)
        if problem is None:
            return True, ""
        return False, (f"{_sentence(problem)}. First wrong piece: "
                       f"{_diagnose_dh(dh_fn, *args)}.")

    def label(i, case):
        return (f"case {i} (a={case['a']:.4g}, alpha={case['alpha']:.4g}, "
                f"d={case['d']:.4g}, theta={case['theta']:.4g})")

    return Grader("ex1: Robot.dh_transform", keep).run(cases, label, check)


# ----------------------------------------------------------------------
# Part (b): forward kinematics
# ----------------------------------------------------------------------


def _diagnose_step(robot, dh, thetas, frames, want, i, tol):
    """Why frame i is wrong when frames 0..i-1 are right: the student's frame
    compared with what each common slip in step i would have produced, all
    built from the student's own dh_transform and the expected frame i - 1."""
    a, alpha, d, offset = dh[i - 1]
    q = thetas[i - 1]
    prev = want[:, :, i - 1]
    step = np.linalg.solve(prev, want[:, :, i])  # the H_i^{i-1} that was needed
    got = frames[:, :, i]

    def dh_(*args):
        return _call_dh(robot.dh_transform, *args)

    try:
        A = dh_(a, alpha, d, q + offset)
    except Exception as err:
        return f"dh_transform raised {_describe(err)} for this joint"
    if not _close(A, step, tol):
        return (f"{_call('dh_transform', [a, alpha, d, q + offset])}, joint {i}'s link, is "
                "itself wrong, so fix part (a) first")
    candidates = [
        (A @ prev, f"the product is in the wrong order: frames[:, :, {i}] = "
                   f"frames[:, :, {i - 1}] @ dh_transform(joint {i}), with the new link on the "
                   "right"),
        (A, f"frames[:, :, {i}] is joint {i}'s dh_transform on its own, not chained on: "
            f"multiply it onto frames[:, :, {i - 1}]"),
        (prev, f"frames[:, :, {i}] is a copy of frame {i - 1}: this joint's link was never "
               "multiplied on"),
    ]
    if abs(offset) > 1e-9:
        candidates.append((prev @ dh_(a, alpha, d, q),
                           f"joint {i}'s theta offset ({offset:.4g} rad) wasn't added to its "
                           "angle: theta_i = q_i + theta_offset_i"))
    candidates.append((prev @ dh_(a, alpha, d, offset),
                       f"joint {i}'s angle q_{i} wasn't used (theta was just the offset): "
                       f"theta_i = q_i + theta_offset_i, with q_i = thetas[{i - 1}]"))
    candidates.append((prev @ dh_(d, alpha, a, q + offset),
                       "a and d are swapped: dh_transform takes (a, alpha, d, theta), and a "
                       "row is [a, alpha, d, theta_offset]"))
    for j in range(len(dh)):
        if j != i - 1:
            r = dh[j]
            candidates.append((prev @ dh_(r[0], r[1], r[2], q + r[3]),
                               f"this used row {j + 1} of the table (dh[{j}]) for joint {i}; "
                               f"joint {i}'s is dh[{i - 1}]"))
    for j in range(len(thetas)):
        if j != i - 1:
            candidates.append((prev @ dh_(a, alpha, d, thetas[j] + offset),
                               f"this used thetas[{j}] (joint {j + 1}'s angle) for joint {i}; "
                               f"joint {i}'s is thetas[{i - 1}]"))
    # More than one slip can produce the same frame on a given case (say, an
    # angle that happens to be 0), so name every one that fits.
    matches = [why for candidate, why in candidates if _close(got, candidate, tol)]
    if not matches:
        return None
    return matches[0] + "".join(f"; or else {why}" for why in matches[1:])


def grade_fk(robot_cls, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        dh = np.array(case["dh"], dtype=float)
        thetas = np.array(case["thetas"], dtype=float)
        n = dh.shape[0]
        robot = robot_cls(dh.copy())
        frames = _array(robot.forward_kinematics(thetas.copy()), "frames", (4, 4, n + 1))
        want = np.asarray(expected["frames"])
        # Positions scale with the arm, so the tolerance does too.
        tol = TOL * max(1.0, float(np.sum(np.abs(dh[:, [0, 2]]))))
        for i in range(n + 1):
            problem = _transform_problem(frames[:, :, i], want[:, :, i], tol)
            if problem is None:
                continue
            if i == 0:
                return False, (f"frames[:, :, 0], the first step of the chain, is wrong: it "
                               f"should be the base frame, the identity. {_sentence(problem)}.")
            where = (f"{_sentence(_right_before(i, ('is right', 'are right')))} and "
                     f"frames[:, :, {i}] is the first wrong one, so the mistake is in step {i} "
                     f"of the chain: frames[:, :, {i}] = frames[:, :, {i - 1}] @ "
                     f"dh_transform(joint {i}), built from joint {i}'s row dh[{i - 1}] and its "
                     f"angle thetas[{i - 1}] + its offset. {_sentence(problem)}")
            try:
                why = _diagnose_step(robot, dh, thetas, frames, want, i, tol)
            except Exception:
                why = None
            return False, where + (f". Likely cause: {why}." if why else ".")
        return True, ""

    def label(i, case):
        return f"case {i} ({case.get('name') or str(len(case['dh'])) + '-joint chain'})"

    return Grader("ex1: Robot.forward_kinematics", keep).run(cases, label, check)


# ----------------------------------------------------------------------
# Local runs: your code, the published fixtures
# ----------------------------------------------------------------------


def _fixture(exercise, k):
    """(case, expected) pairs from exN/inK.txt and exN/outK.txt."""
    return zip(_read_jsonl(ROOT / exercise / f"in{k}.txt"),
               _read_jsonl(ROOT / exercise / f"out{k}.txt"))


def _missing(exercise, k):
    paths = [ROOT / exercise / f"in{k}.txt", ROOT / exercise / f"out{k}.txt"]
    return [str(p) for p in paths if not p.exists()]


_IMPORTED = {}  # relpath -> (module, None) or (None, why it failed): one import per run


def _import_once(relpath):
    if relpath not in _IMPORTED:
        try:
            with _time_limit(IMPORT_TIME_LIMIT, f"importing {relpath}"):
                _IMPORTED[relpath] = (_load_module(ROOT / relpath), None)
        except KeyboardInterrupt:
            raise
        except BaseException as err:  # a syntax error, a hang, or sys.exit() at import
            _IMPORTED[relpath] = (None, _describe(err))
    return _IMPORTED[relpath]


def _load(name, relpath, *attrs):
    """(values, None) for the named attributes of a student file, or
    (None, a Grader that failed saying why)."""
    module, error = _import_once(relpath)
    if module is None:
        failed = Grader(name)
        failed.record(f"{relpath} imports", False, error)
        return None, failed
    missing = [a for a in attrs if not hasattr(module, a)]
    if missing:
        failed = Grader(name)
        failed.record(f"{', '.join(missing)} defined", False, f"not found in {relpath}")
        return None, failed
    return tuple(getattr(module, a) for a in attrs), None


def _run(name, relpath, attrs, exercise, k, grade, keep):
    """Load the student's code, then grade it on exercise/in{k}.txt."""
    missing = _missing(exercise, k)
    if missing:
        failed = Grader(name, keep)
        failed.record("fixtures present", False, "missing " + ", ".join(missing))
        return failed
    loaded, failed = _load(name, relpath, *attrs)
    if failed is not None:
        return failed
    return grade(*loaded, _fixture(exercise, k), keep)


def graders_ex1(keep):
    return [
        _run("ex1: Robot.dh_transform", "ex1/robot.py", ["Robot"], "ex1", 1,
             grade_dh_transform, keep),
        _run("ex1: Robot.forward_kinematics", "ex1/robot.py", ["Robot"], "ex1", 2,
             grade_fk, keep),
    ]


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------


def _meshcat_vis():
    """Import the visualization module, or explain what's missing.

    The graded modes need only numpy, so a student can be grading happily and
    still be one `conda env create` short of the simulator.
    """
    try:
        import meshcat_vis
    except ImportError as err:
        sys.exit(
            f"Visualization needs the course's robot library, and it isn't "
            f"importable: {err}\n"
            "Install it from the copy in this handout:\n"
            "    cd 16384-robot-lib && conda env update -f environment.yml && cd ..\n"
            "    conda activate 16384"
        )
    return meshcat_vis


def visualize_ex1():
    """Pose the simulated xArm7 with sliders; your FK draws its frames over it."""
    module = _load_module(ROOT / "ex1" / "robot.py")
    _meshcat_vis().visualize_ex1(module.Robot, lambda: module.XARM7_DH)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--problem", type=str, default="all",
                        choices=["ex1", "vis1", "all"])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every failure, not just the first few")
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES_SHOWN,
                        help="failures to list per test (default: %(default)s)")
    args = parser.parse_args()

    # The visualization mode shows, it doesn't score; it never reaches the
    # report below.
    if args.problem == "vis1":
        visualize_ex1()
        return

    keep = 10 ** 9 if args.verbose else args.max_failures
    exercises = {"ex1": graders_ex1}
    selected = list(exercises) if args.problem == "all" else [args.problem]

    total_passed = total_count = 0
    for name in selected:
        for grader in exercises[name](keep):
            p, t = grader.report(max_failures=keep)
            total_passed += p
            total_count += t

    print(f"\n=== TOTAL: {total_passed}/{total_count} ===")
    sys.exit(0 if total_passed == total_count else 1)


if __name__ == "__main__":
    main()
