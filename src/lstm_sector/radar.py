"""Polygonal radar ("star") axes, shared by Figures 7 and 8.

Adapted from the matplotlib radar chart example.  `radar_factory` registers a
`radar` projection with one spoke per variable and returns the spoke angles.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars: int, frame: str = 'circle'):
    """Register a radar projection with `num_vars` evenly spaced axes."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Gridlines carry an interpolation step > 1; forcing interpolation
            # stops PolarTransform from turning the polygon into arcs.
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            if frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor='k')
            raise ValueError(f"unknown frame {frame!r}")

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            if frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(
                    Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            raise ValueError(f"unknown frame {frame!r}")

    register_projection(RadarAxes)
    return theta


BASELINE = 0.15   # the worst model still gets a visible radius


def normalise(values: list[float]) -> list[float]:
    """Map a spoke's values onto [BASELINE, 1] so every model stays visible.

    A spoke whose values are all but identical -- annualised volatility once the
    strategies have been levered to a common target, say -- is collapsed to a
    single mid radius. Min-max scaling would otherwise magnify floating-point
    noise into the full range and invent differences that are not there.
    """
    lo, hi = min(values), max(values)
    scale = max(abs(v) for v in values) or 1.0
    if hi - lo <= 1e-6 * scale:
        return [0.5] * len(values)
    return [BASELINE + (1 - BASELINE) * (v - lo) / (hi - lo) for v in values]


def draw(theta, radar_data, model_names, labels, colors, note: str | None = None):
    """Draw one radar chart and return its figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.10 if note else 0.05)

    for data, color, label in zip(radar_data, colors, model_names):
        ax.plot(theta, data, color=color, label=label, linewidth=2.5)
        ax.fill(theta, data, facecolor=color, alpha=0.12)

    ax.set_varlabels([])
    for angle, text in zip(theta, labels):
        # push the side labels outwards so they clear the polygon
        ax.text(angle, 1.08 + 0.10 * abs(np.sin(angle)), text,
                ha='center', va='center', fontsize=16, color='black')

    ax.set_ylim(0, 1)
    ax.set_rgrids([0.25, 0.5, 0.75, 1.0], labels=[''] * 4, angle=0)
    ax.grid(True, color='grey', linestyle='-', linewidth=0.5, alpha=1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.10), fontsize=16)

    if note:
        fig.text(0.5, 0.03, note, ha='center', fontsize=11,
                 style='italic', color='grey')
    return fig
