"""Synthetic 2D phantom and beam dose model (no patient data).

Geometry (voxel units, isocentre at the grid centre, y points up):
  body   : disk, radius 10
  tumour : disk, radius 3, centred on the isocentre
  OAR    : disk, radius 2, centred at OAR_CENTRE (below the tumour)
  normal : body minus tumour minus OAR

Gantry angle theta (degrees, clockwise from the top): the source sits in direction
u = (sin theta, cos theta) and the beam travels along -u through the isocentre.
Each beam is a parallel strip BEAM_WIDTH voxels wide. Inside the strip the dose is
exp(-MU * depth), where depth is the distance from the body surface along the beam.
Each beam is scaled so that it gives DOSE_PER_BEAM at the isocentre, so about
1 / DOSE_PER_BEAM = 4 beams deliver the prescription of 1.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID = 21
BODY_RADIUS = 10.0
TUMOUR_RADIUS = 3.0
OAR_RADIUS = 2.0
OAR_CENTRE = (0.3, -5.7)  # (x, y), 5.7 voxels below the isocentre, slightly off the mirror axis (see README)
BEAM_WIDTH = 7.0          # = tumour diameter + 1 voxel (voxels -3..3 inclusive)
MU = 0.05                 # attenuation per voxel
DOSE_PER_BEAM = 0.25      # dose of one beam at the isocentre


@dataclass
class Phantom:
    n_beams: int
    angles_deg: np.ndarray            # (n_beams,)
    x: np.ndarray                     # (V,) x coordinate of each body voxel
    y: np.ndarray                     # (V,) y coordinate of each body voxel
    rows: np.ndarray                  # (V,) grid row index (for drawing)
    cols: np.ndarray                  # (V,) grid column index
    tumour: np.ndarray                # (V,) bool
    oar: np.ndarray                   # (V,) bool
    normal: np.ndarray                # (V,) bool
    D: np.ndarray                     # (n_beams, V) dose per unit beam weight
    meta: dict = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return self.x.size

    def to_grid(self, values: np.ndarray, fill=np.nan) -> np.ndarray:
        """Scatter a per-voxel vector back onto the GRID x GRID image."""
        img = np.full((GRID, GRID), fill, dtype=float)
        img[self.rows, self.cols] = values
        return img


def beam_dose(theta_deg: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unscaled relative dose of one beam at voxel coordinates (x, y)."""
    t = np.deg2rad(theta_deg)
    u = np.array([np.sin(t), np.cos(t)])       # towards the source
    v = np.array([np.cos(t), -np.sin(t)])      # across the beam
    s = x * u[0] + y * u[1]                    # position along the beam axis
    p = x * v[0] + y * v[1]                    # lateral offset from the axis
    in_strip = np.abs(p) <= BEAM_WIDTH / 2 + 1e-9
    entry = np.sqrt(np.clip(BODY_RADIUS**2 - p**2, 0.0, None))
    depth = np.clip(entry - s, 0.0, None)
    return np.where(in_strip, np.exp(-MU * depth), 0.0)


def build_phantom(n_beams: int = 12, oar_centre: tuple[float, float] = OAR_CENTRE) -> Phantom:
    c = (GRID - 1) / 2
    rr, cc = np.mgrid[0:GRID, 0:GRID]
    xg, yg = cc - c, c - rr
    body = xg**2 + yg**2 <= BODY_RADIUS**2 + 1e-9
    rows, cols = rr[body], cc[body]
    x, y = xg[body].astype(float), yg[body].astype(float)

    tumour = x**2 + y**2 <= TUMOUR_RADIUS**2 + 1e-9
    oar = (x - oar_centre[0])**2 + (y - oar_centre[1])**2 <= OAR_RADIUS**2 + 1e-9
    normal = ~(tumour | oar)

    angles = np.arange(n_beams) * 360.0 / n_beams
    scale = DOSE_PER_BEAM / np.exp(-MU * BODY_RADIUS)   # isocentre depth = body radius
    D = np.stack([scale * beam_dose(a, x, y) for a in angles])

    meta = dict(grid=GRID, body_radius=BODY_RADIUS, tumour_radius=TUMOUR_RADIUS,
                oar_radius=OAR_RADIUS, oar_centre=list(oar_centre), beam_width=BEAM_WIDTH,
                mu=MU, dose_per_beam=DOSE_PER_BEAM, n_voxels=int(x.size),
                n_tumour=int(tumour.sum()), n_oar=int(oar.sum()), n_normal=int(normal.sum()))
    return Phantom(n_beams, angles, x, y, rows, cols, tumour, oar, normal, D, meta)
