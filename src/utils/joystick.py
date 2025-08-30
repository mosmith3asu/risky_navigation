import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import math
import time

class VirtualJoystick:
    """
    Matplotlib-based virtual joystick you can poll in a non-blocking loop.

    Normalized output: (x, y) in [-1, 1].
    - deadzone: radial threshold (normalized units) under which output snaps to (0,0).
    - smoothing: EMA factor in (0,1]; None disables smoothing.
    - spring: if True, knob recenters on mouse release; if False, stays where released.
    """
    def __init__(self, ax=None, radius=1.0, deadzone=0.05, smoothing=None, spring=True):
        self.radius = float(radius)
        self.deadzone = float(deadzone)
        self.smoothing = float(smoothing) if smoothing is not None else None
        self.spring = bool(spring)

        # State
        self.center = (0.0, 0.0)
        self.dragging = False
        self._raw = (0.0, 0.0)      # latest raw (x,y), normalized
        self._value = (0.0, 0.0)    # possibly smoothed (x,y), normalized

        self.invert_x = True

        # Figure / Axes
        if ax is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(4,4))
            self._own_fig = True
        else:
            self.ax = ax
            self.fig = ax.figure
            self._own_fig = False

        # Aesthetics & geometry
        cx, cy = self.center
        self.ax.set_aspect('equal', adjustable='box')
        m = self.radius * 1.2
        self.ax.set_xlim(cx - m, cx + m)
        self.ax.set_ylim(cy - m, cy + m)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Virtual Joystick (drag inside circle)")

        # Patches: boundary, deadzone, knob, vector, crosshairs
        self.boundary = Circle(self.center, self.radius, fill=False, linewidth=2)
        self.dead = Circle(self.center, self.deadzone * self.radius, fill=True, alpha=0.06)
        self.knob_radius = 0.12 * self.radius
        self.knob = Circle(self.center, self.knob_radius, color='C0', alpha=0.9)
        self.vector = Line2D([cx, cx], [cy, cy], linewidth=2, alpha=0.8)

        self.ax.add_patch(self.dead)
        self.ax.add_patch(self.boundary)
        self.ax.add_line(self.vector)
        self.ax.add_patch(self.knob)

        # Crosshairs
        self.ax.add_line(Line2D([cx - self.radius, cx + self.radius], [cy, cy], alpha=0.2))
        self.ax.add_line(Line2D([cx, cx], [cy - self.radius, cy + self.radius], alpha=0.2))

        # Readout text
        self.text = self.ax.text(
            cx, cy - 1.35*self.radius,
            "x=0.00  y=0.00  r=0.00  θ=0.00°",
            ha='center', va='center', fontsize=9
        )

        # Event bindings
        self._cids = []
        self._cids.append(self.fig.canvas.mpl_connect('button_press_event', self._on_press))
        self._cids.append(self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion))
        self._cids.append(self.fig.canvas.mpl_connect('button_release_event', self._on_release))

        if self._own_fig:
            self.fig.tight_layout()
            # non-blocking show
            try:
                self.fig.show()
            except Exception:
                plt.show(block=False)

    # ------------------ Public API ------------------
    def get(self):
        """
        Returns (x, y, r, theta, active)
          x,y   : normalized in [-1,1]
          r     : sqrt(x^2 + y^2) in [0,1]
          theta : angle in radians in [-pi, pi] (atan2)
          active: True if currently dragging
        """
        if self.dragging:
            x, y = self._value
        else:
            x, y = 0, 0

        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        x = -x if  self.invert_x else x  # invert x if needed
        return x, y, r, theta, self.dragging

    def close(self):
        for cid in self._cids:
            self.fig.canvas.mpl_disconnect(cid)
        plt.close(self.fig)

    # ------------------ Event handlers ------------------
    def _inside(self, x, y):
        cx, cy = self.center
        return (x - cx)**2 + (y - cy)**2 <= self.radius**2

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if self._inside(event.xdata, event.ydata):
            self.dragging = True
            self._update_from_point(event.xdata, event.ydata)

    def _on_motion(self, event):
        if not self.dragging:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self._update_from_point(event.xdata, event.ydata)

    def _on_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if self.spring:
            self._set_value((0.0, 0.0))
            self._redraw()

    # ------------------ Helpers ------------------
    def _update_from_point(self, x, y):
        cx, cy = self.center
        dx, dy = x - cx, y - cy
        r = math.hypot(dx, dy)

        # Clamp to boundary
        if r > self.radius and r > 0:
            scale = self.radius / r
            dx *= scale
            dy *= scale
            r = self.radius

        # Normalize to [-1,1]
        xn = dx / self.radius
        yn = dy / self.radius

        # Deadzone
        if math.hypot(xn, yn) < self.deadzone:
            xn, yn = 0.0, 0.0

        self._set_value((xn, yn))
        self._redraw()

    def _set_value(self, xy):
        self._raw = xy
        if self.smoothing is None:
            self._value = xy
        else:
            ax, ay = self._value
            rx, ry = xy
            a = self.smoothing
            self._value = ( (1-a)*ax + a*rx, (1-a)*ay + a*ry )

    def _redraw(self):
        # update graphics from current (possibly smoothed) value
        x, y = self._value
        cx, cy = self.center
        kx, ky = cx + x * self.radius, cy + y * self.radius

        # knob
        self.knob.center = (kx, ky)
        # vector
        self.vector.set_data([cx, kx], [cy, ky])

        # text
        r = math.hypot(x, y)
        theta_deg = math.degrees(math.atan2(y, x)) if r > 0 else 0.0
        self.text.set_text(f"x={x:+.2f}  y={y:+.2f}  r={r:.2f}  θ={theta_deg:+.2f}°")

        # draw
        self.fig.canvas.draw_idle()


# ------------------ Example usage ------------------
if __name__ == "__main__":
    joy = VirtualJoystick(deadzone=0.08, smoothing=0.35, spring=True)

    # Example: poll at ~50 Hz without blocking other work
    try:
        while plt.fignum_exists(joy.fig.number):
            x, y, r, th, active = joy.get()
            # Do anything you like with (x,y) here
            # e.g., send to a controller, update a sim, etc.
            # print(f"{x:+.2f} {y:+.2f} active={active}", end="\r")

            # Keep UI responsive
            plt.pause(0.02)  # ~50 Hz
    finally:
        if plt.fignum_exists(joy.fig.number):
            joy.close()
