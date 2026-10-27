"""Autograder for HW7: 3D transforms and forward kinematics of the xArm7 (ex1).

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


class WrongResult(ValueError):
    """Student code returned something that isn't the array it should be."""


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
        raise WrongResult(f"{name} is None: did the function reach its return statement?")
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as err:
        raise WrongResult(f"{name} isn't an array of numbers ({err})") from None
    if array.shape != tuple(shape):
        raise WrongResult(f"{name} has shape {array.shape}, expected {tuple(shape)}")
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
    if isinstance(err, WrongResult):
        return str(err)
    return f"{type(err).__name__}: {err}"


def _raised(call, err):
    """'f(x) raised ...', or 'f(x) failed: ...' when it returned something unusable."""
    return f"{call} {'failed:' if isinstance(err, WrongResult) else 'raised'} {_describe(err)}"


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
# The tests. Each takes the student's robot.py module and an iterable of
# (case, expected) pairs, so Gradescope's grade.py runs exactly these on its
# perturbed cases.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Part (a): 3D homogeneous transforms
# ----------------------------------------------------------------------

# The elementary transforms, exactly as the writeup's Background section
# prints them. A failing transform() is tried on one parameter at a time,
# against these, to find which factor is wrong.


def _rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=float)


def _rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], dtype=float)


def _rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)


def _translation(x, y, z):
    H = np.eye(4)
    H[:3, 3] = [x, y, z]
    return H


_PARTS = {"rot_x": "(a.1)", "rot_y": "(a.2)", "rot_z": "(a.3)", "translation": "(a.4)",
          "transform": "(a.5)"}
_ELEMENTARY = {"rot_x": _rot_x, "rot_y": _rot_y, "rot_z": _rot_z, "translation": _translation}
_AXES = {"rot_x": "x", "rot_y": "y", "rot_z": "z"}


def _fmt_args(args):
    """'0.3' or '0.1, 0.2, 0.3' or '[0.1, 0.2, 0.3], [0, 0, 0.7]'."""
    return ", ".join(_fmt(a) if np.ndim(a) else f"{a:.4g}" for a in args)


def _call_fn(module, name, args):
    fn = getattr(module, name, None)
    if not callable(fn):
        raise AttributeError(f"robot.py has no function {name}()")
    return _array(fn(*args), f"{name}'s result", (4, 4))


def _elementary_slip(name, args, got, tol):
    """A guess at what went wrong in one of the four elementary functions."""
    if name == "translation":
        if _close(got.T, _translation(*args), tol):
            return "the offset is in the bottom row; it goes in the last column, H[:3, 3]"
        if _close(got[:3, :3], np.zeros((3, 3)), tol):
            return "the rotation block is zero; with no rotation it is the identity"
        return None
    (theta,) = args
    if _close(got, _ELEMENTARY[name](-theta), tol):
        return ("it rotates by -theta: the two sin terms have each other's signs"
                + (" (about y the + sin is in the top row, [0, 2], and the - sin in the"
                   " third, [2, 0])" if name == "rot_y" else ""))
    for other, axis in _AXES.items():
        if other != name and _close(got, _ELEMENTARY[other](theta), tol):
            return f"it is a rotation about {axis}, not {_AXES[name]}"
    return None


def _check_elementary(module, name, args, tol):
    """None if the student's elementary function is right on `args`, else why not."""
    try:
        got = _call_fn(module, name, args)
    except WrongResult as err:
        return f"{name}({_fmt_args(args)}), part {_PARTS[name]}, is wrong: {err}"
    except Exception as err:
        return _raised(f"{name}({_fmt_args(args)})", err)
    problem = _transform_problem(got, _ELEMENTARY[name](*args), tol)
    if problem is None:
        return None
    slip = _elementary_slip(name, args, got, tol)
    return (f"{name}({_fmt_args(args)}), part {_PARTS[name]}, is wrong: {problem}"
            + (f". Likely cause: {slip}" if slip else ""))


def _diagnose_transform(module, xyz, rpy, got, tol):
    """Which piece of Trans(x, y, z) Rot_z(yaw) Rot_y(pitch) Rot_x(roll) is wrong."""
    roll, pitch, yaw = rpy
    # Their own building blocks first, on this case's numbers.
    for name, args in [("translation", tuple(xyz)), ("rot_z", (yaw,)), ("rot_y", (pitch,)),
                       ("rot_x", (roll,))]:
        problem = _check_elementary(module, name, args, tol)
        if problem is not None:
            return f"{problem}, and transform() is built from it, so fix that first"
    zero = (0.0, 0.0, 0.0)
    pieces = [
        ("the translation", (tuple(xyz), zero), _translation(*xyz),
         "a translation by xyz", "translation(x, y, z)"),
        ("the yaw", (zero, (0.0, 0.0, yaw)), _rot_z(yaw), "a rotation by yaw about z",
         "rot_z(yaw)"),
        ("the pitch", (zero, (0.0, pitch, 0.0)), _rot_y(pitch), "a rotation by pitch about y",
         "rot_y(pitch)"),
        ("the roll", (zero, (roll, 0.0, 0.0)), _rot_x(roll), "a rotation by roll about x",
         "rot_x(roll)"),
    ]
    for what, args, want, meaning, call in pieces:
        try:
            alone = _call_fn(module, "transform", args)
        except Exception as err:
            return _raised(f"transform({_fmt_args(args)})", err)
        problem = _transform_problem(alone, want, tol)
        if problem is not None:
            angle = max(args[1], key=abs)
            mixed = [f"; it is a rotation about {axis}: {what[4:]} is going into {other}(), "
                     f"not {call[:5]}()" for other, axis in _AXES.items()
                     if angle and _close(alone, _ELEMENTARY[other](angle), tol)]
            return (f"{what}. With every other parameter 0, transform({_fmt_args(args)}) "
                    f"should be just {meaning}, {call}, but {problem}" + "".join(mixed[:1]))
    # Each piece is right on its own, so the order is wrong. Which order?
    T, Rz, Ry, Rx = _translation(*xyz), _rot_z(yaw), _rot_y(pitch), _rot_x(roll)
    orders = [
        (T @ Rx @ Ry @ Rz, "the three rotations are in reverse order (roll first)"),
        (Rz @ Ry @ Rx @ T, "the translation is on the right, after the rotations"),
        (Rx @ Ry @ Rz @ T, "the whole product is reversed"),
    ]
    matches = [why for H, why in orders if _close(got, H, tol)]
    why = "; or else ".join(matches) if matches else "they are multiplied in the wrong order"
    return (f"none on its own: each piece is right with the other parameters 0, but {why}. "
            "H = Trans(x, y, z) Rot_z(yaw) Rot_y(pitch) Rot_x(roll), left to right")


def grade_transforms(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        name, args = case["fn"], case["args"]
        want = np.asarray(expected["H"])
        if name in _ELEMENTARY:
            problem = _check_elementary(module, name, tuple(args), TOL)
            return problem is None, f"{problem}."  # starts with the function's name: no capital
        H = _call_fn(module, "transform", args)
        problem = _transform_problem(H, want, TOL)
        if problem is None:
            return True, ""
        return False, (f"{_sentence(problem)}. First wrong piece: "
                       f"{_diagnose_transform(module, args[0], args[1], H, TOL)}.")

    def label(i, case):
        return f"case {i}: {case['fn']}({_fmt_args(case['args'])}), part {_PARTS[case['fn']]}"

    return Grader("ex1 (a): rot_x, rot_y, rot_z, translation, transform", keep).run(
        cases, label, check)


# ----------------------------------------------------------------------
# Part (b): forward kinematics
# ----------------------------------------------------------------------


def _diagnose_step(module, joints, thetas, frames, want, i, tol):
    """Why frame i is wrong when frames 0..i-1 are right: the student's frame
    compared with what each common slip in step i would have produced, all
    built from the student's own transform() and rot_z() and the expected
    frame i - 1."""
    row = joints[i - 1]
    xyz, rpy = row[:3], row[3:]
    q = thetas[i - 1]
    prev = want[:, :, i - 1]
    step = np.linalg.solve(prev, want[:, :, i])  # the link transform that was needed
    got = frames[:, :, i]

    def tf(*args):
        return _call_fn(module, "transform", args)

    def rot(axis, angle):
        return _call_fn(module, f"rot_{axis}", (angle,))

    try:
        offset = tf(xyz, rpy)
        link = offset @ rot("z", q)
    except Exception as err:
        return _raised("your part (a) functions, for this joint,", err)
    if not _close(link, step, tol):
        return (f"transform({_fmt_args([xyz, rpy])}) @ rot_z({q:.4g}), joint {i}'s link, is "
                "itself wrong, so fix part (a) first")
    candidates = [
        (link @ prev, f"the product is in the wrong order: frames[:, :, {i}] = "
                      f"frames[:, :, {i - 1}] @ link, with the new link on the right"),
        (prev @ rot("z", q) @ offset,
         f"the joint's rotation comes before its offset: link = transform(xyz, rpy) @ "
         f"rot_z(q_{i}), the offset first"),
        (link, f"frames[:, :, {i}] is joint {i}'s link on its own, not chained on: multiply "
               f"it onto frames[:, :, {i - 1}]"),
        (prev, f"frames[:, :, {i}] is a copy of frame {i - 1}: this joint's link was never "
               "multiplied on"),
        (prev @ offset, f"joint {i}'s angle wasn't used: the link ends with rot_z(q_{i}), "
                        f"with q_{i} = thetas[{i - 1}]"),
        (prev @ rot("z", q), f"joint {i}'s offset transform(xyz, rpy), from row joints[{i - 1}], "
                             "wasn't applied"),
        (prev @ tf(xyz, (0.0, 0.0, 0.0)) @ rot("z", q),
         f"joint {i}'s orientation rpy = row[3:] was left out"),
        (prev @ offset @ rot("x", q), f"joint {i} turns about x here; every joint turns about "
                                      "its frame's z axis, rot_z"),
        (prev @ offset @ rot("y", q), f"joint {i} turns about y here; every joint turns about "
                                      "its frame's z axis, rot_z"),
        (prev @ tf(rpy, xyz) @ rot("z", q),
         "xyz and rpy are swapped: a row is [x, y, z, roll, pitch, yaw], so xyz = row[:3] and "
         "rpy = row[3:]"),
        (prev @ tf(xyz, rpy[::-1]) @ rot("z", q),
         "the angles were read as (yaw, pitch, roll); a row's are (roll, pitch, yaw)"),
    ]
    for j in range(len(joints)):
        if j != i - 1:
            r = joints[j]
            candidates.append((prev @ tf(r[:3], r[3:]) @ rot("z", q),
                               f"this used row {j + 1} of the table (joints[{j}]) for joint {i}; "
                               f"joint {i}'s is joints[{i - 1}]"))
    for j in range(len(thetas)):
        if j != i - 1:
            candidates.append((prev @ offset @ rot("z", thetas[j]),
                               f"this used thetas[{j}] (joint {j + 1}'s angle) for joint {i}; "
                               f"joint {i}'s is thetas[{i - 1}]"))
    # More than one slip can produce the same frame on a given case (say, an
    # angle that happens to be 0), so name every one that fits.
    matches = [why for candidate, why in candidates if _close(got, candidate, tol)]
    if not matches:
        return None
    return matches[0] + "".join(f"; or else {why}" for why in matches[1:])


def grade_fk(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        joints = np.array(case["joints"], dtype=float)
        thetas = np.array(case["thetas"], dtype=float)
        n = joints.shape[0]
        robot = module.Robot(joints.copy())
        frames = _array(robot.forward_kinematics(thetas.copy()), "frames", (4, 4, n + 1))
        want = np.asarray(expected["frames"])
        # Positions scale with the arm, so the tolerance does too.
        tol = TOL * max(1.0, float(np.sum(np.abs(joints[:, :3]))))
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
                     f"transform(xyz, rpy) @ rot_z(q_{i}), with joint {i}'s row "
                     f"joints[{i - 1}] and its angle thetas[{i - 1}]. {_sentence(problem)}")
            try:
                why = _diagnose_step(module, joints, thetas, frames, want, i, tol)
            except Exception:
                why = None
            return False, where + (f". Likely cause: {why}." if why else ".")
        return True, ""

    def label(i, case):
        return f"case {i} ({case.get('name') or str(len(case['joints'])) + '-joint chain'})"

    return Grader("ex1 (b): Robot.forward_kinematics", keep).run(cases, label, check)


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
    """(the student's module, None), or (None, a Grader that failed saying
    why) if it doesn't import or lacks one of `attrs`."""
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
    return module, None


def _run(name, relpath, attrs, exercise, k, grade, keep):
    """Load the student's code, then grade it on exercise/in{k}.txt."""
    missing = _missing(exercise, k)
    if missing:
        failed = Grader(name, keep)
        failed.record("fixtures present", False, "missing " + ", ".join(missing))
        return failed
    module, failed = _load(name, relpath, *attrs)
    if failed is not None:
        return failed
    return grade(module, _fixture(exercise, k), keep)


TRANSFORMS = ["rot_x", "rot_y", "rot_z", "translation", "transform"]


def graders_ex1(keep):
    return [
        _run("ex1 (a): rot_x, rot_y, rot_z, translation, transform", "ex1/robot.py",
             TRANSFORMS, "ex1", 1, grade_transforms, keep),
        _run("ex1 (b): Robot.forward_kinematics", "ex1/robot.py", ["Robot"] + TRANSFORMS,
             "ex1", 2, grade_fk, keep),
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
    _meshcat_vis().visualize_ex1(module.Robot, lambda: module.XARM7_JOINTS)


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
