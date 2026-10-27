"""Run the students' HW7 forward kinematics on the simulated xArm7, in meshcat.

Sliders pose the arm; the simulator moves it. Your `forward_kinematics`, on
the xArm7's DH table, is handed the joint angles the arm is actually at, and
draws every one of its frames over the arm, with a line through their origins. Two checks run
against the simulator's own model of the arm, which is where the DH frames
come from:

    the end effector   your last frame against the flange (the `link7` body)
    every frame        frame i's z axis should run along joint i+1's axis, its
                       positive way, with its origin on that axis (that is what
                       DH frames are)

Nothing in here grades anything. `local_autograder.py` decides right and wrong
from the fixtures; this only shows you what your code does.
"""

import collections
import http.server
import json
import math
import socket
import string
import threading
import time
import urllib.parse
import webbrowser

import meshcat.geometry as g
import numpy as np
from xarm7_lib import HOME_POSE, SimulatedXArm7
from xarm7_lib.mujoco_meshcat import MeshcatVisualizer

# Everything this module draws lives under here, clear of the "mujoco" and
# "safety" prefixes `MeshcatVisualizer` publishes under.
PREFIX = "fk"

_ARM_COLOR = 0xFFC53D  # your frames' origins, and the line through them
_JOINT_MARKER_RADIUS = 0.012
_TRIAD_SCALE = 0.08
_EE_TRIAD_SCALE = 0.14

# How far off a frame may be before the status line says so.
_POSITION_OK = 0.002  # m
_ANGLE_OK = math.radians(1.0)

# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _node(viz, path):
    return viz.viewer[f"{PREFIX}/{path}"]


def draw_polyline(viz, path, points, color=_ARM_COLOR, width=3.0):
    """Draw an open polyline through `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=np.float32).T  # meshcat wants (3, n)
    with viz.lock:
        _node(viz, path).set_object(
            g.Line(
                g.PointsGeometry(points),
                g.LineBasicMaterial(color=color, linewidth=width),
            )
        )


# Marker sets that have already been published, so a redraw only has to move
# them. These are drawn every frame, and re-sending the spheres each time would
# be a dozen ZMQ round trips per frame, taken against the same viewer lock the
# simulation thread needs to sync.
_MARKERS = {}


def draw_markers(viz, path, points, radius=_JOINT_MARKER_RADIUS, color=_ARM_COLOR):
    """Draw a small sphere at each of `points`, an (n, 3) array."""
    points = np.asarray(points, dtype=float)
    key = (id(viz), path)
    shape = (len(points), float(radius), int(color))

    with viz.lock:
        if _MARKERS.get(key) != shape:
            _node(viz, path).delete()
            material = g.MeshLambertMaterial(color=color)
            for i in range(len(points)):
                _node(viz, f"{path}/{i}").set_object(g.Sphere(radius), material)
            _MARKERS[key] = shape
        for i, point in enumerate(points):
            T = np.eye(4)
            T[:3, 3] = point
            _node(viz, f"{path}/{i}").set_transform(T)


def draw_triad(viz, path, T, scale=_TRIAD_SCALE):
    """Draw an RGB axis triad at the 4x4 pose `T`."""
    with viz.lock:
        node = _node(viz, path)
        node.set_object(g.triad(scale))
        node.set_transform(np.asarray(T, dtype=float))


def clear(viz, path=""):
    """Remove everything this module drew under `path`."""
    with viz.lock:
        node = _node(viz, path) if path else viz.viewer[PREFIX]
        node.delete()
    for key in [k for k in _MARKERS if k[0] == id(viz) and k[1].startswith(path)]:
        del _MARKERS[key]


# ----------------------------------------------------------------------
# The slider page
# ----------------------------------------------------------------------

# `string.Template` rather than `str.format`: the page is mostly CSS and JS, and
# `.format` would need every brace in it doubled.
_PAGE = string.Template("""<!doctype html>
<title>$title</title>
<style>
  body { margin: 0; font: 14px system-ui, sans-serif; background: #14171c; color: #e8ecf1; }
  #panel { padding: 10px 16px 12px; border-bottom: 1px solid #2b313a; }
  .row { display: flex; align-items: center; gap: 12px; margin: 6px 0; }
  .row label { width: 5.5em; color: #9aa7b8; }
  .row input { flex: 1; }
  .row output { width: 7em; text-align: right; font-variant-numeric: tabular-nums; }
  #status { margin-top: 8px; color: #9aa7b8; font-variant-numeric: tabular-nums; }
  iframe { display: block; width: 100vw; height: calc(100vh - $chrome); border: 0; }
  button { background: #2b313a; color: #e8ecf1; border: 0; padding: 4px 12px;
           border-radius: 4px; cursor: pointer; }
</style>
<div id="panel">
$rows
  <div class="row">
    <div id="status">&mdash;</div>
    <button id="reset">reset</button>
  </div>
</div>
<iframe src="$meshcat_url"></iframe>
<script>
  const SPECS = $specs;
  const el = k => document.getElementById(k);

  function show(spec) {
    const v = (+el("s_" + spec.key).value).toFixed(spec.digits);
    el("o_" + spec.key).textContent = v + spec.unit;
  }
  function push() {
    SPECS.forEach(show);
    fetch("set?" + SPECS.map(s => s.key + "=" + el("s_" + s.key).value).join("&"));
  }
  SPECS.forEach(s => el("s_" + s.key).addEventListener("input", push));
  el("reset").addEventListener("click", () => {
    SPECS.forEach(s => el("s_" + s.key).value = s.value);
    push();
  });

  setInterval(async () => {
    const state = await (await fetch("state")).json();
    el("status").innerHTML = state.status;
  }, 250);
  push();
</script>
""")

_ROW = string.Template(
    '  <div class="row">\n'
    '    <label for="s_$key">$label</label>\n'
    '    <input id="s_$key" type="range" min="$low" max="$high" step="$step" value="$value">\n'
    '    <output id="o_$key"></output>\n'
    "  </div>"
)

# Page chrome above the viewer: one row per slider, plus the status row.
_ROW_HEIGHT = 30  # px
_PANEL_PADDING = 40  # px


class SliderSpec(collections.namedtuple(
    "SliderSpec", "key label low high step value unit digits"
)):
    """One slider on the page.

    `unit` and `digits` are only how the browser renders the number back to the
    reader; python always works in the raw value.
    """

    __slots__ = ()

    def __new__(cls, key, label, low, high, step, value, unit="", digits=2):
        return super().__new__(
            cls, key, label, float(low), float(high), float(step),
            float(np.clip(value, low, high)), unit, int(digits),
        )


class SliderPage:
    """Sliders on a local page, with the meshcat viewer embedded below.

    meshcat's own dat.GUI can't be driven from python — its ZMQ bridge forwards
    only the five scene commands and drops anything else, and the browser->python
    direction is wired up for image captures alone — so the controls are served
    from here instead, and the viewer is put in an iframe so it is still one tab.

    Values live behind a lock: `values` is what the caller reads each tick, and
    `set_status` is the line the page echoes back. The page knows nothing about
    what the numbers mean; formatting the status line is the caller's job.
    """

    def __init__(self, specs, meshcat_url, title="xArm7", open_browser=True):
        self._specs = list(specs)
        self._lock = threading.Lock()
        self._values = {s.key: s.value for s in self._specs}
        self._status = "&mdash;"

        page = _PAGE.substitute(
            title=title,
            meshcat_url=meshcat_url,
            chrome=f"{_PANEL_PADDING + _ROW_HEIGHT * (len(self._specs) + 1)}px",
            rows="\n".join(
                _ROW.substitute(
                    key=s.key, label=s.label,
                    low=f"{s.low:.4f}", high=f"{s.high:.4f}",
                    step=f"{s.step:g}", value=f"{s.value:.4f}",
                )
                for s in self._specs
            ),
            specs=json.dumps([
                {"key": s.key, "value": s.value, "unit": s.unit, "digits": s.digits}
                for s in self._specs
            ]),
        ).encode("utf-8")

        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", _free_port()), _handler_for(self, page)
        )
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="fk-sliders", daemon=True
        )
        self._thread.start()

        host, port = self._server.server_address
        self.url = f"http://{host}:{port}/"
        print(f"[vis] controls: {self.url}")
        if open_browser:
            webbrowser.open(self.url)

    @property
    def values(self):
        """The current slider values, by key."""
        with self._lock:
            return dict(self._values)

    def set_status(self, text):
        """Publish the line the page shows under the sliders."""
        with self._lock:
            self._status = str(text)

    def _accept(self, query):
        parsed = urllib.parse.parse_qs(query)
        with self._lock:
            for spec in self._specs:
                value = parsed.get(spec.key)
                if value:
                    self._values[spec.key] = float(np.clip(
                        float(value[0]), spec.low, spec.high
                    ))

    def _state(self):
        with self._lock:
            return {"values": dict(self._values), "status": self._status}

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _handler_for(sliders, page):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            route, _, query = self.path.lstrip("/").partition("?")
            if route in ("", "index.html"):
                self._reply("text/html; charset=utf-8", page)
            elif route == "set":
                sliders._accept(query)
                self._reply("text/plain", b"ok")
            elif route == "state":
                body = json.dumps(sliders._state()).encode("utf-8")
                self._reply("application/json", body)
            else:
                self.send_error(404)

        def _reply(self, content_type, body):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # a slider drag is a request per frame; don't narrate it

    return Handler


# ----------------------------------------------------------------------
# Your code, and what to do when it isn't ready yet
# ----------------------------------------------------------------------


class Complaints:
    """Say once, per distinct message, that the student's code isn't usable yet.

    Every demo calls into the student's code every frame, and the stubs raise
    or return zeros until the To-Dos are filled in. The arm keeps moving
    regardless; this only keeps the terminal readable.
    """

    def __init__(self, what, where):
        self.what = what
        self.where = where
        self._last = None

    def __call__(self, err):
        message = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
        if message == self._last:
            return
        self._last = message
        print(f"[vis] {self.what} isn't usable yet — {message}\n"
              f"      The arm still moves; fill in the To-Dos in {self.where} "
              "to see your answer drawn.")

    def clear(self):
        self._last = None


def _checked(value, name, shape):
    """`value` as a float array of `shape`; refuses None, NaN and all-zero stubs."""
    if value is None:
        raise ValueError(f"{name} is None")
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} has NaN or inf in it")
    if not np.any(array):
        raise ValueError(f"{name} is all zeros (still the stub?)")
    return array


def _serve(rate, tick, wait=time.sleep):
    """Call `tick()` at `rate` Hz until ctrl-c. `wait` passes the time between
    ticks: a sleep, or a matplotlib pause that keeps its window responsive."""
    period = 1.0 / float(rate)
    while True:
        tick()
        wait(period)



# ----------------------------------------------------------------------
# ex1: forward kinematics, on sliders
# ----------------------------------------------------------------------


def _slider_range(arm, index):
    """A joint's range, trimmed to one turn: joints 1, 3, 5 and 7 can turn
    through two, and a slider across both re-traces the same poses."""
    low, high = (float(v) for v in arm.position_limit[index])
    if high - low > 2.0 * math.pi:
        middle = 0.5 * (low + high)
        low, high = middle - math.pi, middle + math.pi
    return low, high


def _angle_between(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(1.0, cos)))


def _rotation_angle(R):
    """The angle of the rotation R, in rad."""
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def _truth(arm):
    """(flange pose, [(axis, anchor) of joints 1..7]) from the simulator, in its
    world frame. Read under the arm's lock: the simulation thread writes it."""
    with arm._lock:
        body = arm.model.body("link7").id
        flange = np.eye(4)
        flange[:3, :3] = arm.data.xmat[body].reshape(3, 3)
        flange[:3, 3] = arm.data.xpos[body]
        axes = [(arm.data.xaxis[j].copy(), arm.data.xanchor[j].copy()) for j in arm._jid]
        q = arm.data.qpos[arm._qadr].copy()
    return q, flange, axes


def check_frames(world, flange, axes):
    """What the simulator thinks of your frames: (flange error mm, deg, first problem)."""
    ee = world[-1]
    position = float(np.linalg.norm(ee[:3, 3] - flange[:3, 3]))
    angle = _rotation_angle(ee[:3, :3].T @ flange[:3, :3])
    for i, (axis, anchor) in enumerate(axes):
        z, origin = world[i][:3, 2], world[i][:3, 3]
        # DH puts z_i along joint i+1's axis, and pointing its positive way:
        # the way the joint turns for a positive angle.
        tilt = _angle_between(z, axis)
        off_axis = float(np.linalg.norm(np.cross(origin - anchor, axis)))
        if math.pi - tilt < _ANGLE_OK:
            return position, angle, (f"frame {i}'s z axis points the opposite way to "
                                     f"joint {i + 1}'s")
        if tilt > _ANGLE_OK:
            return position, angle, (f"frame {i}'s z axis is {math.degrees(tilt):.0f} deg "
                                      f"off joint {i + 1}'s axis")
        if off_axis > _POSITION_OK:
            return position, angle, (f"frame {i}'s origin is {off_axis * 1000:.0f} mm "
                                     f"off joint {i + 1}'s axis")
    return position, angle, None


class SkeletonPlot:
    """Your forward kinematics as a stick figure, in a matplotlib 3D window.

    Drawn in the robot's base frame (z = 0 is the mounting surface), from your
    frames alone: a line from each frame's origin to the next, a dot at each,
    and the end effector's axes in red, green and blue for x, y and z. A green
    cross marks where the simulator's flange really is, so a correct end
    effector sits on it. The window opens beside the browser tab and follows
    the sliders. Closing it doesn't stop the demo.
    """

    _LIMITS = ((-0.75, 0.75), (-0.75, 0.75), (0.0, 1.1))  # m, around the arm's reach
    _TRIAD = 0.1  # m

    def __init__(self):
        import matplotlib
        import matplotlib.pyplot as plt

        self.plt = plt
        self.fig = None
        if matplotlib.get_backend().lower() in ("agg", "pdf", "ps", "svg", "cairo", "template"):
            print("[vis] no display for the matplotlib window (backend "
                  f"{matplotlib.get_backend()}); the browser view still works.")
            return
        plt.ion()
        self.fig = plt.figure("Your forward kinematics - xArm7", figsize=(6.5, 6.5))
        ax = self.ax = self.fig.add_subplot(projection="3d")
        (self.links,) = ax.plot([], [], [], "-", color="#1f4fa8", lw=3, label="your links")
        (self.joints,) = ax.plot([], [], [], "o", color="#1f4fa8", ms=6,
                                 label="your frame origins")
        self.ee = [ax.plot([], [], [], "-", color=c, lw=2.5, label=label)[0]
                   for c, label in (("#d62728", "your end effector: x (red), y, z"),
                                    ("#2ca02c", None), ("#1f77b4", None))]
        (self.flange,) = ax.plot([], [], [], "x", color="#2ca02c", ms=12, mew=3,
                                 label="the simulator's flange")
        (self.base,) = ax.plot([0], [0], [0], "s", color="0.3", ms=8, label="base")
        for setter, (low, high) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), self._LIMITS):
            setter(low, high)
        try:  # matplotlib >= 3.6 can zoom the box in
            ax.set_box_aspect([high - low for low, high in self._LIMITS], zoom=1.15)
        except TypeError:
            ax.set_box_aspect([high - low for low, high in self._LIMITS])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.view_init(elev=20, azim=-60)
        ax.legend(loc="upper left", fontsize=8)
        self.title = self.fig.suptitle("", fontsize=9)
        self.fig.tight_layout(rect=(0, 0, 1, 0.92))
        self.plt.show(block=False)

    @property
    def open(self):
        return self.fig is not None and self.plt.fignum_exists(self.fig.number)

    def update(self, frames=None, flange=None, status=""):
        """Redraw from `frames` (4, 4, n + 1), in the base frame; None clears."""
        if not self.open:
            return
        if frames is None:
            for line in [self.links, self.joints, *self.ee]:
                line.set_data_3d([], [], [])
        else:
            origins = frames[:3, 3, :]
            self.links.set_data_3d(*origins)
            self.joints.set_data_3d(*origins)
            tip, R = frames[:3, 3, -1], frames[:3, :3, -1]
            for k, line in enumerate(self.ee):
                end = tip + self._TRIAD * R[:, k]
                line.set_data_3d(*np.column_stack([tip, end]))
        if flange is not None:
            self.flange.set_data_3d(*np.asarray(flange, dtype=float).reshape(3, 1))
        self.title.set_text(status)
        self.fig.canvas.draw_idle()

    def wait(self, seconds):
        """Let the window redraw and respond for `seconds`, or just sleep."""
        if self.open:
            self.plt.pause(seconds)
        else:
            time.sleep(seconds)


def _plain(html):
    """The status line without its HTML, for the plot's title."""
    import re
    return re.sub(r"<[^>]+>", "", html).replace("&nbsp;", " ")


def visualize_ex1(robot_cls, dh_table_fn, rate=30.0, open_browser=True):
    """Pose the arm with sliders; your forward_kinematics draws its frames over it.

    `dh_table_fn` returns the xArm7's DH table (XARM7_DH, from robot.py): the
    frames are those of that table through your forward_kinematics.

    The arm is driven by the simulator, not by your code, so the demo works
    before any of it does. Your frames are drawn in the simulator's world
    frame: the arm's base frame sits 12 cm up there, on the model's mount. The
    same frames are also drawn as a stick figure in a matplotlib window beside
    it (see `SkeletonPlot`), in the base frame.
    """
    # The safety box is off: the sliders reach poses outside the lab's box, and
    # this is only a picture. Self-collision checking stays on.
    arm = SimulatedXArm7(visualize=False, safety_box=None, q0=np.array(HOME_POSE))
    page = None
    try:
        arm.viz = MeshcatVisualizer(arm.model, arm.data, open_browser=False)
        print(f"[vis] meshcat: {arm.viz.url}")
        base = arm._base_frame()
        page = SliderPage(
            [SliderSpec(f"q{i}", f"joint{i + 1}", *_slider_range(arm, i), 0.005,
                        HOME_POSE[i], " rad", 2) for i in range(7)],
            arm.viz.url,
            title="Forward kinematics — xArm7",
            open_browser=open_browser,
        )
        complaints = Complaints("Robot.forward_kinematics", "ex1/robot.py")
        plot = SkeletonPlot()
        base_inv = np.linalg.inv(base)
        print("[vis] drag the sliders; ctrl-c here to stop.")

        def tick():
            values = page.values
            arm.servo_joints(np.array([values[f"q{i}"] for i in range(7)]))
            q, flange, axes = _truth(arm)

            try:
                table = _checked(dh_table_fn(), "XARM7_DH", (7, 4))
                robot = robot_cls(table)
                frames = _checked(robot.forward_kinematics(q.copy()), "frames", (4, 4, 8))
                world = [base @ frames[:, :, i] for i in range(8)]
                origins = np.array([T[:3, 3] for T in world])
                draw_polyline(arm.viz, "links", origins)
                draw_markers(arm.viz, "joints", origins)
                for i, T in enumerate(world):
                    draw_triad(arm.viz, f"frames/{i}", T,
                               scale=_EE_TRIAD_SCALE if i == 7 else _TRIAD_SCALE)
                complaints.clear()
            except (Exception, SystemExit) as err:  # the demo outlives your code
                complaints(err)
                clear(arm.viz)
                status = ("your forward_kinematics isn't answering yet "
                          "(see the terminal)")
                page.set_status(status)
                plot.update(None, (base_inv @ flange)[:3, 3], status)
                return

            position, angle, problem = check_frames(world, flange, axes)
            verdict = ("<b style='color:#7bd88f'>every frame on its joint axis</b>"
                       if problem is None else f"<b style='color:#ff5a5f'>{problem}</b>")
            status = (f"your end effector vs the flange: <b>{position * 1000:.1f}</b> mm, "
                      f"<b>{math.degrees(angle):.1f}</b> deg &nbsp; {verdict}")
            page.set_status(status)
            plot.update(frames, (base_inv @ flange)[:3, 3],
                        _plain(status).replace("deg   ", "deg\n"))

        _serve(rate, tick, wait=plot.wait)
    except KeyboardInterrupt:
        print("\n[vis] stopping.")
    finally:
        if page is not None:
            page.close()
        arm.stop(wait=False)
        arm.close()
