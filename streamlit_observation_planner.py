#!/usr/bin/env python3
"""
4MOST Observation Block Planner

=============================

Author
------
Dr. Vivek M.
Indian Institute of Astrophysics (IIA)
Bangalore, India
Email: vivek.m@iiap.res.in

Version
-------
v1.5

Description
-----------
4MOST Observation Block Planner  is an interactive scheduling tool designed
to assist in planning and optimizing astronomical observing campaigns.
The software ingests observing requests from multiple surveys and generates
optimized nightly schedules while respecting physical observing constraints.

The planner evaluates the observability of each target based on:

    • Airmass constraints
    • Sky brightness conditions
    • Moon separation constraints
    • Twilight boundaries
    • Target visibility

Only targets that satisfy these physical feasibility conditions are considered
for scheduling. Among the feasible targets, the scheduler computes a weighted
score that balances:

    • Survey fairness
    • Scientific priority
    • Target visibility
    • Observational urgency

The observation with the highest score is scheduled for each available time
slot.

Outputs produced by the planner include:

    • Nightly observing schedules
    • Campaign observing plans
    • Survey completion summaries
    • Progress history across nights
    • Risk diagnostics identifying difficult-to-observe targets

The software is implemented as a Streamlit application and uses the Astropy
and Astroplan ecosystems for astronomical calculations.

Dependencies
------------
Python >= 3.9

Required Python packages:

    streamlit
    pandas
    numpy
    astropy
    astroplan
    matplotlib
    plotly
    reportlab

Install dependencies using:

    pip install streamlit pandas numpy astropy astroplan matplotlib plotly reportlab

Usage
-----
Launch the application using:

    streamlit run streamlit_observation_planner_app.py

This will open a browser-based interface where the user can upload survey
target lists, configure scheduling parameters, and generate observing plans.

License
-------
For academic and research use.

Notes
-----
The scheduling algorithm separates physical feasibility constraints from
optimization criteria. Hard constraints such as airmass limits, sky conditions,
and Moon separation are applied first. Only targets that satisfy these
conditions are passed to the weighted scoring model used for scheduling.
"""

from __future__ import annotations

import io
import math
import os
import time
import zipfile
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from astroplan import Observer
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("ob_scheduler")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def log_event(message: str, logs: Optional[List[str]] = None) -> None:
        logger.info(message)
        if logs is not None:
            logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


class timed:
    def __init__(self, label: str, logs: Optional[List[str]], stats: Dict[str, float]):
        self.label = label
        self.logs = logs
        self.stats = stats
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        log_event(f"Started: {self.label}", self.logs)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self.t0
        self.stats[self.label] = self.stats.get(self.label, 0.0) + dt
        log_event(f"Finished: {self.label} in {dt:.2f} s", self.logs)


# -----------------------------------------------------------------------------
# Defaults and dataclasses
# -----------------------------------------------------------------------------
DEFAULT_OBSERVATORY = {
    "name": "ESO Paranal",
    "latitude_deg": -24.6667,
    "longitude_deg": -70.4167,
    "elevation_m": 2635.0,
    "timezone": "America/Santiago",
    "temperature_c": 10.0,
    "pressure_hpa": 743.0,
    "relative_humidity": 0.15,
}
DEFAULT_GRID_MIN = 5
DEFAULT_OVERHEAD_S = 120
DEFAULT_MOON_ILLUM_DARK_MAX = 0.25
DEFAULT_MOON_ILLUM_GREY_MAX = 0.65
DEFAULT_MOON_ALT_BRIGHT_MIN_DEG = 0.0


@dataclass
class ObservatoryConfig:
    name: str
    latitude_deg: float
    longitude_deg: float
    elevation_m: float
    timezone: str
    temperature_c: float = 10.0
    pressure_hpa: float = 743.0
    relative_humidity: float = 0.15

    def location(self) -> EarthLocation:
        return EarthLocation(
            lat=self.latitude_deg * u.deg,
            lon=self.longitude_deg * u.deg,
            height=self.elevation_m * u.m,
        )

    def observer(self) -> Observer:
        return Observer(
            location=self.location(),
            name=self.name,
            timezone=self.timezone,
            temperature=self.temperature_c * u.deg_C,
            pressure=self.pressure_hpa * u.hPa,
            relative_humidity=self.relative_humidity,
        )


@dataclass
class OBRequest:
    survey: str
    ob_name: str
    ra_deg: float
    dec_deg: float
    airmass_min: float
    airmass_max: float
    exp_time_s: float
    overhead_s: float
    priority: int
    sky_class: str
    nexp: int
    min_moon_sep_angle: float
    already_completed_obs: int

    @property
    def required_total_s(self) -> float:
        return self.nexp * (self.exp_time_s + self.overhead_s)


@dataclass
class ExposureBlock:
    survey: str
    ob_name: str
    block_id: str
    ra_deg: float
    dec_deg: float
    airmass_min: float
    airmass_max: float
    exp_time_s: float
    overhead_s: float
    priority: int
    sky_class: str
    min_moon_sep_angle: float
    scheduled: bool = False

    @property
    def total_duration_s(self) -> float:
        return self.exp_time_s + self.overhead_s

    @property
    def coord(self) -> SkyCoord:
        return SkyCoord(self.ra_deg * u.deg, self.dec_deg * u.deg, frame="icrs")


@dataclass
class ScheduledBlock:
    survey: str
    ob_name: str
    start_local: datetime
    end_local: datetime
    ra_deg: float
    dec_deg: float
    exp_time_s: float
    overhead_s: float
    airmass_start: float
    airmass_mid: float
    moon_sep_deg: float
    sky_required: str
    sky_actual: str
    priority: int
    urgency_score: float
    date_beyond_unobservable: Optional[date]
    observable_nights_count: int


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def section_title(title: str, bg: str = "#eef4ff") -> None:
    st.markdown(
        f"""
        <div style="background:{bg}; padding:10px 14px; border-radius:10px; border:1px solid #d6e3ff; margin:8px 0 10px 0;">
            <span style="font-size:1.05rem; font-weight:700; color:#1f2d3d;">{title}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_run_config(
    obs: "ObservatoryConfig",
    planning_mode: str,
    start_date: date,
    end_date: date,
    overhead_s: float,
    grid_minutes: int,
    urgency_grid_minutes: int,
    include_risk_diagnostics: bool,
    dark_illum_max: float,
    grey_illum_max: float,
    bright_alt_min_deg: float,
    fairness_enabled: bool,
    fairness_weight: float,
    priority_enabled: bool,
    priority_weight: float,
    visibility_enabled: bool,
    visibility_weight: float,
    urgency_enabled: bool,
    urgency_weight: float,
    output_name: str,
    code_filename: str,
) -> Dict[str, str]:
    return {
        "Observatory name": obs.name,
        "Latitude [deg]": f"{obs.latitude_deg:.6f}",
        "Longitude [deg]": f"{obs.longitude_deg:.6f}",
        "Elevation [m]": f"{obs.elevation_m:.1f}",
        "Timezone": obs.timezone,
        "Temperature [C]": f"{obs.temperature_c:.1f}",
        "Pressure [hPa]": f"{obs.pressure_hpa:.1f}",
        "Relative humidity": f"{obs.relative_humidity:.2f}",
        "Planning mode": planning_mode,
        "Start date": str(start_date),
        "End date": str(end_date),
        "Overhead per exposure [s]": f"{overhead_s:.1f}",
        "Scheduler grid [min]": str(grid_minutes),
        "Urgency/Risk grid [min]": str(urgency_grid_minutes),
        "Include risk diagnostics": str(include_risk_diagnostics),
        "Dark illumination max": f"{dark_illum_max:.2f}",
        "Grey illumination max": f"{grey_illum_max:.2f}",
        "Bright altitude threshold [deg]": f"{bright_alt_min_deg:.1f}",
        "Fairness enabled": str(fairness_enabled),
        "Fairness weight": f"{fairness_weight:.2f}" if fairness_enabled else "0.00",
        "Priority enabled": str(priority_enabled),
        "Priority weight": f"{priority_weight:.2f}" if priority_enabled else "0.00",
        "Visibility enabled": str(visibility_enabled),
        "Visibility weight": f"{visibility_weight:.2f}" if visibility_enabled else "0.00",
        "Urgency enabled": str(urgency_enabled),
        "Urgency weight": f"{urgency_weight:.2f}" if urgency_enabled else "0.00",
        "Generated at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Code filename": code_filename,
        "Output name": output_name,
    }


def config_table_for_pdf(run_config: Dict[str, str]) -> Table:
    rows = [["Parameter", "Value"]] + [[k, v] for k, v in run_config.items()]
    tbl = Table(rows, repeatRows=1, colWidths=[8.0 * cm, 18.0 * cm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


# -----------------------------------------------------------------------------
# Parsing and normalization
# -----------------------------------------------------------------------------
def normalize_sky_class(value: str) -> str:
    v = str(value).strip().lower()
    if v in {"gray", "grey"}:
        return "grey"
    if v not in {"dark", "grey", "bright", "any"}:
        return "any"
    return v


def parse_airmass_min(v) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x < 0:
        return 1.0
    return float(x)


def parse_airmass_max(v) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x < 0:
        return 99.0
    return float(x)


def parse_moon_sep(v) -> float:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x) or x < 0:
        return 0.0
    return float(x)


def parse_ob_files(uploaded_files, overhead_s: float, logs: Optional[List[str]] = None) -> pd.DataFrame:
    frames = []
    log_event(f"Parsing {len(uploaded_files)} uploaded files", logs)
    for up in uploaded_files:
        df = pd.read_csv(up)
        df["source_file"] = up.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    required = [
        "survey", "ob_name", "ra_deg", "dec_deg", "airmass_min", "airmass_max",
        "exp_time_s", "priority", "sky_class", "nexp", "min_moon_sep_angle",
        "already_completed_obs",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["survey"] = out["survey"].astype(str)
    out["ob_name"] = out["ob_name"].astype(str)
    out["ra_deg"] = pd.to_numeric(out["ra_deg"], errors="coerce")
    out["dec_deg"] = pd.to_numeric(out["dec_deg"], errors="coerce")
    out["exp_time_s"] = pd.to_numeric(out["exp_time_s"], errors="coerce").fillna(0.0)
    out["priority"] = pd.to_numeric(out["priority"], errors="coerce").fillna(3).astype(int)
    out["nexp"] = pd.to_numeric(out["nexp"], errors="coerce").fillna(1).astype(int)
    out["already_completed_obs"] = pd.to_numeric(out["already_completed_obs"], errors="coerce").fillna(0).astype(int)
    out["airmass_min"] = out["airmass_min"].map(parse_airmass_min)
    out["airmass_max"] = out["airmass_max"].map(parse_airmass_max)
    out["min_moon_sep_angle"] = out["min_moon_sep_angle"].map(parse_moon_sep)
    out["sky_class"] = out["sky_class"].map(normalize_sky_class)
    out["overhead_s"] = float(overhead_s)
    out = out.dropna(subset=["ra_deg", "dec_deg"]).reset_index(drop=True)
    log_event(f"Merged uploaded files into {len(out)} rows", logs)
    return out


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def airmass_from_alt_deg(alt_deg: float) -> float:
    z_deg = 90.0 - alt_deg
    if z_deg >= 89.0:
        return np.inf
    return float(1.0 / np.cos(np.deg2rad(z_deg)))


def moon_illumination_fraction(t: Time) -> float:
    sun = get_sun(t)
    moon = get_body("moon", t)
    phase_angle = sun.separation(moon).rad
    return 0.5 * (1.0 - np.cos(phase_angle))


def classify_sky_time(moon_alt_deg: float, moon_illum: float, dark_illum_max: float, grey_illum_max: float, bright_alt_min_deg: float) -> str:
    if moon_alt_deg < bright_alt_min_deg:
        return "dark"
    if moon_illum <= dark_illum_max:
        return "dark"
    if moon_illum <= grey_illum_max:
        return "grey"
    return "bright"


def sky_class_preference_rank(requested: str, actual: str) -> Optional[int]:
    requested = normalize_sky_class(requested)
    actual = normalize_sky_class(actual)
    rank_map = {
        "dark": {"dark": 0, "grey": 1, "bright": 2, "any": 2},
        "grey": {"grey": 0, "bright": 1, "any": 1},
        "bright": {"bright": 0, "any": 1},
        "any": {"any": 0, "bright": 0, "grey": 0, "dark": 0},
    }
    return rank_map.get(actual, {}).get(requested, None)


# -----------------------------------------------------------------------------
# Expansion and grouping
# -----------------------------------------------------------------------------
def expand_blocks(df: pd.DataFrame, logs: Optional[List[str]] = None) -> Tuple[List[ExposureBlock], pd.DataFrame]:
    requests: List[OBRequest] = []
    blocks: List[ExposureBlock] = []
    for _, row in df.iterrows():
        req = OBRequest(
            survey=str(row["survey"]),
            ob_name=str(row["ob_name"]),
            ra_deg=float(row["ra_deg"]),
            dec_deg=float(row["dec_deg"]),
            airmass_min=float(row["airmass_min"]),
            airmass_max=float(row["airmass_max"]),
            exp_time_s=float(row["exp_time_s"]),
            overhead_s=float(row["overhead_s"]),
            priority=int(row["priority"]),
            sky_class=str(row["sky_class"]),
            nexp=max(0, int(row["nexp"])),
            min_moon_sep_angle=float(row["min_moon_sep_angle"]),
            already_completed_obs=max(0, int(row["already_completed_obs"])),
        )
        requests.append(req)
        remaining = max(0, req.nexp - req.already_completed_obs)
        for j in range(remaining):
            blocks.append(
                ExposureBlock(
                    survey=req.survey,
                    ob_name=req.ob_name,
                    block_id=f"{req.ob_name}_exp{j+1}",
                    ra_deg=req.ra_deg,
                    dec_deg=req.dec_deg,
                    airmass_min=req.airmass_min,
                    airmass_max=req.airmass_max,
                    exp_time_s=req.exp_time_s,
                    overhead_s=req.overhead_s,
                    priority=req.priority,
                    sky_class=req.sky_class,
                    min_moon_sep_angle=req.min_moon_sep_angle,
                )
            )

    req_df = pd.DataFrame([
        {
            "survey": r.survey,
            "required_total_s": r.required_total_s,
            "already_completed_s": min(r.already_completed_obs, r.nexp) * (r.exp_time_s + r.overhead_s),
        }
        for r in requests
    ])
    survey_rollup = req_df.groupby("survey", as_index=False).agg(
        required_total_s=("required_total_s", "sum"),
        already_completed_s=("already_completed_s", "sum"),
    )
    log_event(f"Expanded input to {len(blocks)} schedulable exposure blocks over {len(survey_rollup)} surveys", logs)
    return blocks, survey_rollup


def urgency_parent_key(block: ExposureBlock) -> Tuple:
    return (
        block.survey,
        block.ob_name,
        round(block.ra_deg, 8),
        round(block.dec_deg, 8),
        round(block.airmass_min, 4),
        round(block.airmass_max, 4),
        block.sky_class,
        round(block.min_moon_sep_angle, 4),
    )


# -----------------------------------------------------------------------------
# Night geometry and observability caches
# -----------------------------------------------------------------------------
def compute_night_boundaries(obs: ObservatoryConfig, night_date: date) -> Dict[str, datetime]:
    observer = obs.observer()
    tz = ZoneInfo(obs.timezone)
    noon_local = datetime(night_date.year, night_date.month, night_date.day, 12, 0, 0, tzinfo=tz)
    t0 = Time(noon_local)
    sunset = observer.sun_set_time(t0, which="next").to_datetime(timezone=tz)
    eve_civil = observer.twilight_evening_civil(t0, which="next").to_datetime(timezone=tz)
    eve_naut = observer.twilight_evening_nautical(t0, which="next").to_datetime(timezone=tz)
    eve_astro = observer.twilight_evening_astronomical(t0, which="next").to_datetime(timezone=tz)
    tref = Time(sunset + timedelta(hours=6))
    morn_astro = observer.twilight_morning_astronomical(tref, which="next").to_datetime(timezone=tz)
    morn_naut = observer.twilight_morning_nautical(tref, which="next").to_datetime(timezone=tz)
    morn_civil = observer.twilight_morning_civil(tref, which="next").to_datetime(timezone=tz)
    sunrise = observer.sun_rise_time(t0, which="next").to_datetime(timezone=tz)
    return {
        "sunset": sunset,
        "sunrise": sunrise,
        "evening_civil": eve_civil,
        "evening_nautical": eve_naut,
        "evening_astronomical": eve_astro,
        "morning_astronomical": morn_astro,
        "morning_nautical": morn_naut,
        "morning_civil": morn_civil,
    }


def build_night_grid(obs: ObservatoryConfig, night_date: date, grid_minutes: int) -> pd.DataFrame:
    bounds = compute_night_boundaries(obs, night_date)
    start = bounds["evening_astronomical"]
    end = bounds["morning_astronomical"]
    times_local = []
    t = start
    while t < end:
        times_local.append(t)
        t += timedelta(minutes=grid_minutes)
    if not times_local:
        return pd.DataFrame()
    atimes = Time(times_local)
    frame = AltAz(obstime=atimes, location=obs.location())
    moon = get_body("moon", atimes)
    moon_alt = moon.transform_to(frame).alt.deg
    illum = np.array([moon_illumination_fraction(tt) for tt in atimes])
    df = pd.DataFrame({"time_local": times_local, "moon_alt_deg": moon_alt, "moon_illum": illum})
    df.attrs["boundaries"] = bounds
    return df


def add_sky_classification(night_grid: pd.DataFrame, dark_illum_max: float, grey_illum_max: float, bright_alt_min_deg: float) -> pd.DataFrame:
    ng = night_grid.copy()
    ng["sky_class"] = [
        classify_sky_time(ma, mi, dark_illum_max, grey_illum_max, bright_alt_min_deg)
        for ma, mi in zip(ng["moon_alt_deg"], ng["moon_illum"])
    ]
    return ng


def target_altaz_series(obs: ObservatoryConfig, coord: SkyCoord, times_local: List[datetime]) -> AltAz:
    atimes = Time(times_local)
    frame = AltAz(obstime=atimes, location=obs.location())
    return coord.transform_to(frame)


def moon_sep_series_deg(coord: SkyCoord, times_local: List[datetime], location: EarthLocation) -> np.ndarray:
    atimes = Time(times_local)
    moon = get_body("moon", atimes)
    return moon.separation(coord).deg


def block_feasible_at_index(block: ExposureBlock, idx: int, night_grid: pd.DataFrame, target_airmass: np.ndarray, moon_sep_deg: np.ndarray, grid_minutes: int) -> Tuple[bool, Optional[Dict[str, float]]]:
    nslots = math.ceil(block.total_duration_s / (grid_minutes * 60.0))
    if idx + nslots > len(night_grid):
        return False, None
    sub = night_grid.iloc[idx: idx + nslots]
    am_sub = target_airmass[idx: idx + nslots]
    ms_sub = moon_sep_deg[idx: idx + nslots]
    if np.any(~np.isfinite(am_sub)):
        return False, None
    if np.any(am_sub < block.airmass_min) or np.any(am_sub > block.airmass_max):
        return False, None
    sky_ranks = []
    for actual in sub["sky_class"].values:
        rank = sky_class_preference_rank(block.sky_class, actual)
        if rank is None:
            return False, None
        sky_ranks.append(rank)
    if np.any(ms_sub < block.min_moon_sep_angle):
        return False, None
    return True, {
        "nslots": nslots,
        "airmass_start": float(am_sub[0]),
        "airmass_mid": float(am_sub[len(am_sub) // 2]),
        "moon_sep_mid": float(ms_sub[len(ms_sub) // 2]),
        "sky_actual": str(sub["sky_class"].iloc[0]),
        "sky_rank": int(max(sky_ranks)) if sky_ranks else 99,
    }


def precompute_urgency_observability_matrix(
    obs: ObservatoryConfig,
    blocks: List[ExposureBlock],
    start_date: date,
    end_date: date,
    urgency_grid_minutes: int,
    dark_illum_max: float,
    grey_illum_max: float,
    bright_alt_min_deg: float,
    logs: Optional[List[str]] = None,
) -> Tuple[List[date], Dict[Tuple, np.ndarray]]:
    nights = list(daterange(start_date, end_date))
    if not nights or not blocks:
        return nights, {}

    log_event(
        f"Precomputing urgency observability matrix for {len(blocks)} exposure blocks on {len(nights)} nights using {urgency_grid_minutes}-min grid",
        logs,
    )

    unique_by_parent: Dict[Tuple, ExposureBlock] = {}
    for b in blocks:
        unique_by_parent.setdefault(urgency_parent_key(b), b)

    parent_keys = list(unique_by_parent.keys())
    parent_blocks = [unique_by_parent[k] for k in parent_keys]
    observable: Dict[Tuple, np.ndarray] = {k: np.zeros(len(nights), dtype=bool) for k in parent_keys}
    location = obs.location()

    for ni, night in enumerate(nights):
        ng = build_night_grid(obs, night, urgency_grid_minutes)
        if ng.empty:
            continue
        ng = add_sky_classification(ng, dark_illum_max, grey_illum_max, bright_alt_min_deg)
        times_local = list(ng["time_local"])
        atimes = Time(times_local)
        frame = AltAz(obstime=atimes, location=location)
        moon = get_body("moon", atimes)

        target_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
        for pb in parent_blocks:
            tkey = (round(pb.ra_deg, 8), round(pb.dec_deg, 8))
            if tkey not in target_cache:
                coord = pb.coord
                alt_deg = coord.transform_to(frame).alt.deg
                target_cache[tkey] = {
                    "airmass": np.array([airmass_from_alt_deg(a) for a in alt_deg]),
                    "moon_sep": moon.separation(coord).deg,
                }

        for pkey, pb in zip(parent_keys, parent_blocks):
            tkey = (round(pb.ra_deg, 8), round(pb.dec_deg, 8))
            cache = target_cache[tkey]
            feasible_any = False
            for idx in range(len(ng)):
                feasible, _ = block_feasible_at_index(pb, idx, ng, cache["airmass"], cache["moon_sep"], urgency_grid_minutes)
                if feasible:
                    feasible_any = True
                    break
            observable[pkey][ni] = feasible_any

    log_event("Finished urgency observability matrix", logs)
    return nights, observable


def precompute_nightly_urgency_cache(
    nights: List[date],
    observability_matrix: Dict[Tuple, np.ndarray],
    logs: Optional[List[str]] = None,
) -> Dict[Tuple[Tuple, date], Tuple[float, Optional[date], int]]:
    cache: Dict[Tuple[Tuple, date], Tuple[float, Optional[date], int]] = {}
    if not nights:
        return cache
    night_index = {n: i for i, n in enumerate(nights)}
    log_event("Building nightly urgency cache from observability matrix", logs)
    for pkey, arr in observability_matrix.items():
        for night in nights:
            i = night_index[night]
            tail = arr[i:]
            n_obs = int(np.sum(tail))
            if n_obs == 0:
                cache[(pkey, night)] = (-np.inf, None, 0)
            else:
                true_inds = np.where(tail)[0]
                last_idx = i + int(true_inds[-1])
                last_obs = nights[last_idx]
                days_left = max((last_obs - night).days, 0)
                beyond = last_obs + timedelta(days=1)
                urgency = (1.0 / max(n_obs, 1)) + (1.0 / max(days_left + 1, 1))
                cache[(pkey, night)] = (float(urgency), beyond, n_obs)
    log_event("Finished nightly urgency cache", logs)
    return cache


# -----------------------------------------------------------------------------
# Risk diagnostics
# -----------------------------------------------------------------------------
def compute_block_risk_table(
    blocks: List[ExposureBlock],
    start_date: date,
    nights: List[date],
    observability_matrix: Dict[Tuple, np.ndarray],
    logs: Optional[List[str]] = None,
) -> pd.DataFrame:
    rows = []
    unique_parents: Dict[Tuple, ExposureBlock] = {}
    for b in blocks:
        unique_parents.setdefault(urgency_parent_key(b), b)

    for pkey, pb in unique_parents.items():
        arr = observability_matrix.get(pkey, np.zeros(len(nights), dtype=bool))
        n_obs = int(np.sum(arr))
        if n_obs == 0:
            risk_quotient = 1e9
            beyond_unobservable = start_date
        else:
            last_idx = int(np.where(arr)[0][-1])
            last_obs = nights[last_idx]
            days_left = max((last_obs - start_date).days, 0)
            beyond_unobservable = last_obs + timedelta(days=1)
            risk_quotient = (1.0 / max(n_obs, 1)) + (1.0 / max(days_left + 1, 1))
        rows.append(
            {
                "survey": pb.survey,
                "ob_name": pb.ob_name,
                "ra_deg": pb.ra_deg,
                "dec_deg": pb.dec_deg,
                "risk_quotient": float(risk_quotient),
                "observable_nights": n_obs,
                "date_beyond_unobservable": beyond_unobservable,
            }
        )
    risk_df = pd.DataFrame(rows)
    if not risk_df.empty:
        risk_df = risk_df.sort_values(["risk_quotient", "date_beyond_unobservable", "observable_nights"], ascending=[False, True, True]).reset_index(drop=True)
        risk_df["risk_rank"] = np.arange(1, len(risk_df) + 1)
    log_event(f"Computed risk diagnostics for {len(unique_parents)} parent OBs", logs)
    return risk_df


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------
def compute_visibility_score(airmass_mid: float) -> float:
    return -1e9 if not np.isfinite(airmass_mid) else max(0.0, 2.5 - airmass_mid)


def compute_priority_score(priority: int, max_priority: int) -> float:
    return float(max_priority - priority + 1)


def compute_fairness_score(survey: str, survey_required: Dict[str, float], survey_observed: Dict[str, float]) -> float:
    req = survey_required.get(survey, 0.0)
    obs = survey_observed.get(survey, 0.0)
    frac = 0.0 if req <= 0 else obs / req
    return 1.0 - frac


# -----------------------------------------------------------------------------
# Best OBs at given UT
# -----------------------------------------------------------------------------
def best_obs_at_given_ut(
    obs: ObservatoryConfig,
    df: pd.DataFrame,
    ut_time_str: str,
    ref_date: date,
    dark_illum_max: float,
    grey_illum_max: float,
    bright_alt_min_deg: float,
) -> pd.DataFrame:
    blocks, _ = expand_blocks(df)
    if not blocks:
        return pd.DataFrame()
    try:
        parts = [int(x) for x in ut_time_str.strip().split(":")]
        hh, mm = parts[:2]
        ss = parts[2] if len(parts) == 3 else 0
    except Exception as exc:
        raise ValueError("UT time must be in HH:MM or HH:MM:SS format") from exc

    ut_dt = datetime(ref_date.year, ref_date.month, ref_date.day, hh, mm, ss, tzinfo=ZoneInfo("UTC"))
    local_dt = ut_dt.astimezone(ZoneInfo(obs.timezone))
    at = Time(local_dt)
    frame = AltAz(obstime=at, location=obs.location())
    moon = get_body("moon", at)
    moon_alt = moon.transform_to(frame).alt.deg
    moon_illum = moon_illumination_fraction(at)
    actual_sky = classify_sky_time(moon_alt, moon_illum, dark_illum_max, grey_illum_max, bright_alt_min_deg)

    rows = []
    for b in blocks:
        alt = b.coord.transform_to(frame).alt.deg
        am = airmass_from_alt_deg(float(alt))
        moon_sep = moon.separation(b.coord).deg
        if not np.isfinite(am) or am < b.airmass_min or am > b.airmass_max:
            continue
        sky_rank = sky_class_preference_rank(b.sky_class, actual_sky)
        if sky_rank is None:
            continue
        if moon_sep < b.min_moon_sep_angle:
            continue
        rows.append(
            {
                "survey": b.survey,
                "ob_name": b.ob_name,
                "ra_deg": b.ra_deg,
                "dec_deg": b.dec_deg,
                "priority": b.priority,
                "altitude_deg": float(alt),
                "airmass": float(am),
                "moon_sep_deg": float(moon_sep),
                "sky_actual": actual_sky,
                "sky_required": b.sky_class,
                "sky_rank": int(sky_rank),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["priority", "sky_rank", "airmass", "altitude_deg"], ascending=[True, True, True, False]).head(5).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Nightly scheduling and campaign scheduling
# -----------------------------------------------------------------------------
def build_progress_snapshot(survey_required: Dict[str, float], survey_observed: Dict[str, float], night_date: date) -> pd.DataFrame:
    rows = []
    for survey, req in survey_required.items():
        obs_s = survey_observed.get(survey, 0.0)
        frac = obs_s / req if req > 0 else 0.0
        rows.append(
            {
                "night": pd.Timestamp(night_date),
                "survey": survey,
                "required_total_hr": req / 3600.0,
                "observed_total_hr": obs_s / 3600.0,
                "completion_fraction": frac,
            }
        )
    return pd.DataFrame(rows)


def schedule_single_night(
    obs: ObservatoryConfig,
    night_date: date,
    blocks: List[ExposureBlock],
    survey_required: Dict[str, float],
    survey_observed: Dict[str, float],
    scheduler_grid_minutes: int,
    urgency_cache: Dict[Tuple[Tuple, date], Tuple[float, Optional[date], int]],
    fairness_weight: float,
    priority_weight: float,
    visibility_weight: float,
    urgency_weight: float,
    dark_illum_max: float,
    grey_illum_max: float,
    bright_alt_min_deg: float,
) -> Tuple[List[ScheduledBlock], pd.DataFrame, Dict[str, datetime]]:
    night_grid = build_night_grid(obs, night_date, scheduler_grid_minutes)
    if night_grid.empty:
        return [], pd.DataFrame(), {}
    night_grid = add_sky_classification(night_grid, dark_illum_max, grey_illum_max, bright_alt_min_deg)
    bounds = night_grid.attrs["boundaries"]
    times_local = list(night_grid["time_local"])

    target_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}
    max_priority = max([b.priority for b in blocks], default=1)
    location = obs.location()
    atimes = Time(times_local)
    frame = AltAz(obstime=atimes, location=location)
    moon = get_body("moon", atimes)
    for b in blocks:
        tkey = (round(b.ra_deg, 8), round(b.dec_deg, 8))
        if tkey not in target_cache:
            coord = b.coord
            alt_deg = coord.transform_to(frame).alt.deg
            target_cache[tkey] = {
                "airmass": np.array([airmass_from_alt_deg(a) for a in alt_deg]),
                "moon_sep": moon.separation(coord).deg,
            }

    scheduled: List[ScheduledBlock] = []
    occupied = np.zeros(len(night_grid), dtype=bool)
    idx = 0
    while idx < len(night_grid):
        if occupied[idx]:
            idx += 1
            continue

        candidates = []
        for b in blocks:
            if b.scheduled:
                continue
            tkey = (round(b.ra_deg, 8), round(b.dec_deg, 8))
            cache = target_cache[tkey]
            feasible, meta = block_feasible_at_index(b, idx, night_grid, cache["airmass"], cache["moon_sep"], scheduler_grid_minutes)
            if not feasible or meta is None:
                continue
            pkey = urgency_parent_key(b)
            urgency_score, date_beyond_unobservable, observable_nights_count = urgency_cache.get((pkey, night_date), (-np.inf, None, 0))
            if observable_nights_count == 0 or not np.isfinite(urgency_score):
                continue
            total_score = (
                fairness_weight * compute_fairness_score(b.survey, survey_required, survey_observed)
                + priority_weight * compute_priority_score(b.priority, max_priority)
                + visibility_weight * compute_visibility_score(meta["airmass_mid"])
                + urgency_weight * urgency_score
            )
            meta["urgency_score"] = urgency_score
            meta["date_beyond_unobservable"] = date_beyond_unobservable
            meta["observable_nights_count"] = observable_nights_count
            candidates.append((total_score, b, meta))

        if not candidates:
            idx += 1
            continue

        best_sky_rank = min(int(c[2].get("sky_rank", 99)) for c in candidates)
        candidates = [c for c in candidates if int(c[2].get("sky_rank", 99)) == best_sky_rank]
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, chosen, meta = candidates[0]

        nslots = int(meta["nslots"])
        start_local = times_local[idx]
        end_local = start_local + timedelta(seconds=chosen.total_duration_s)
        scheduled.append(
            ScheduledBlock(
                survey=chosen.survey,
                ob_name=chosen.ob_name,
                start_local=start_local,
                end_local=end_local,
                ra_deg=chosen.ra_deg,
                dec_deg=chosen.dec_deg,
                exp_time_s=chosen.exp_time_s,
                overhead_s=chosen.overhead_s,
                airmass_start=float(meta["airmass_start"]),
                airmass_mid=float(meta["airmass_mid"]),
                moon_sep_deg=float(meta["moon_sep_mid"]),
                sky_required=chosen.sky_class,
                sky_actual=str(meta["sky_actual"]),
                priority=chosen.priority,
                urgency_score=float(meta["urgency_score"]),
                date_beyond_unobservable=meta["date_beyond_unobservable"],
                observable_nights_count=int(meta["observable_nights_count"]),
            )
        )
        chosen.scheduled = True
        survey_observed[chosen.survey] = survey_observed.get(chosen.survey, 0.0) + chosen.total_duration_s
        occupied[idx: idx + nslots] = True
        idx += nslots

    schedule_df = pd.DataFrame([
        {
            "start_local": s.start_local,
            "end_local": s.end_local,
            "survey": s.survey,
            "ob_name": s.ob_name,
            "ra_deg": s.ra_deg,
            "dec_deg": s.dec_deg,
            "exp_time_s": s.exp_time_s,
            "overhead_s": s.overhead_s,
            "airmass_start": s.airmass_start,
            "airmass_mid": s.airmass_mid,
            "moon_sep_deg": s.moon_sep_deg,
            "sky_required": s.sky_required,
            "sky_actual": s.sky_actual,
            "priority": s.priority,
            "urgency_score": s.urgency_score,
            "date_beyond_unobservable": s.date_beyond_unobservable,
            "observable_nights_count": s.observable_nights_count,
        }
        for s in scheduled
    ])
    return scheduled, schedule_df, bounds


def schedule_campaign(
    obs: ObservatoryConfig,
    start_date: date,
    end_date: date,
    blocks: List[ExposureBlock],
    survey_required_df: pd.DataFrame,
    grid_minutes: int,
    urgency_grid_minutes: int,
    fairness_weight: float,
    priority_weight: float,
    visibility_weight: float,
    urgency_weight: float,
    dark_illum_max: float,
    grey_illum_max: float,
    bright_alt_min_deg: float,
    logs: Optional[List[str]] = None,
    stats: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[date, pd.DataFrame], Dict[date, Dict[str, datetime]], pd.DataFrame, pd.DataFrame, List[date], Dict[Tuple, np.ndarray]]:
    nights = list(daterange(start_date, end_date))
    log_event(
        f"Starting campaign scheduling from {start_date} to {end_date} over {len(nights)} nights using scheduler grid={grid_minutes} min and urgency/risk grid={urgency_grid_minutes} min",
        logs,
    )
    survey_required = dict(zip(survey_required_df["survey"], survey_required_df["required_total_s"]))
    survey_observed = dict(zip(survey_required_df["survey"], survey_required_df["already_completed_s"]))

    if stats is None:
        stats = {}

    with timed("precompute_urgency_observability_matrix", logs, stats):
        urgency_nights, observability_matrix = precompute_urgency_observability_matrix(
            obs, blocks, start_date, end_date, urgency_grid_minutes,
            dark_illum_max, grey_illum_max, bright_alt_min_deg, logs,
        )
    with timed("precompute_nightly_urgency_cache", logs, stats):
        urgency_cache = precompute_nightly_urgency_cache(urgency_nights, observability_matrix, logs)

    nightly_tables: Dict[date, pd.DataFrame] = {}
    nightly_bounds: Dict[date, Dict[str, datetime]] = {}
    progress_history: List[pd.DataFrame] = []

    for night in nights:
        t0_night = time.perf_counter()
        _, schedule_df, bounds = schedule_single_night(
            obs=obs,
            night_date=night,
            blocks=blocks,
            survey_required=survey_required,
            survey_observed=survey_observed,
            scheduler_grid_minutes=grid_minutes,
            urgency_cache=urgency_cache,
            fairness_weight=fairness_weight,
            priority_weight=priority_weight,
            visibility_weight=visibility_weight,
            urgency_weight=urgency_weight,
            dark_illum_max=dark_illum_max,
            grey_illum_max=grey_illum_max,
            bright_alt_min_deg=bright_alt_min_deg,
        )
        nightly_tables[night] = schedule_df
        nightly_bounds[night] = bounds
        progress_history.append(build_progress_snapshot(survey_required, survey_observed, night))
        log_event(f"Scheduled night {night}: {len(schedule_df)} OBs in {time.perf_counter() - t0_night:.2f} s", logs)

    summary_df = pd.DataFrame([
        {
            "survey": survey,
            "required_total_hr": req / 3600.0,
            "observed_total_hr": survey_observed.get(survey, 0.0) / 3600.0,
            "completion_fraction": (survey_observed.get(survey, 0.0) / req) if req > 0 else 0.0,
        }
        for survey, req in survey_required.items()
    ]).sort_values("completion_fraction")
    progress_df = pd.concat(progress_history, ignore_index=True) if progress_history else pd.DataFrame()
    return nightly_tables, nightly_bounds, summary_df, progress_df, urgency_nights, observability_matrix


# -----------------------------------------------------------------------------
# PDF helpers
# -----------------------------------------------------------------------------
def make_progress_plot_bytes(progress_df: pd.DataFrame, title: str = "Survey completion fraction vs time") -> bytes:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig, ax = plt.subplots(figsize=(11, 6))
    if progress_df is not None and not progress_df.empty:
        pdf = progress_df.copy()
        pdf["night"] = pd.to_datetime(pdf["night"])
        for survey, grp in pdf.groupby("survey"):
            grp = grp.sort_values("night")
            ax.plot(grp["night"], grp["completion_fraction"], linewidth=1.2, alpha=0.9, label=str(survey))
        ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Completion fraction")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def altitude_airmass_plot_bytes(obs: ObservatoryConfig, night_date: date, bounds: Dict[str, datetime], schedule_df: pd.DataFrame, grid_minutes: int = 5) -> bytes:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    buf = io.BytesIO()
    start = bounds.get("evening_astronomical")
    end = bounds.get("morning_astronomical")
    eve_civil = bounds.get("evening_civil")
    eve_naut = bounds.get("evening_nautical")
    eve_astro = bounds.get("evening_astronomical")
    morn_astro = bounds.get("morning_astronomical")
    morn_naut = bounds.get("morning_nautical")
    morn_civil = bounds.get("morning_civil")

    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax2 = ax.twinx()
    if start is None or end is None:
        ax.text(0.5, 0.5, "No twilight boundaries available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    times_local = []
    t = eve_civil if eve_civil is not None else start
    tend = morn_civil if morn_civil is not None else end
    while t <= tend:
        times_local.append(t)
        t += timedelta(minutes=grid_minutes)
    atimes = Time(times_local)
    frame = AltAz(obstime=atimes, location=obs.location())
    times_ut = [tt.astimezone(ZoneInfo("UTC")) for tt in times_local]
    moon_alt = get_body("moon", atimes).transform_to(frame).alt.deg

    twilight_specs = [
        (eve_civil, eve_naut, "lightgrey", "Civil", "black", 86),
        (eve_naut, eve_astro, "grey", "Nautical", "white", 82),
        (eve_astro, start, "black", "Astronomical", "white", 78),
        (end, morn_astro, "black", "Astronomical", "white", 78),
        (morn_astro, morn_naut, "grey", "Nautical", "white", 82),
        (morn_naut, morn_civil, "lightgrey", "Civil", "black", 86),
    ]
    for x0, x1, c, lab, tc, y in twilight_specs:
        if x0 and x1:
            x0u, x1u = x0.astimezone(ZoneInfo("UTC")), x1.astimezone(ZoneInfo("UTC"))
            ax.axvspan(x0u, x1u, color=c, alpha=0.5)
            ax.text(x0u + (x1u - x0u) / 2, y, lab, ha="center", va="center", fontsize=9, color=tc)

    ax.set_xlim(times_ut[0], times_ut[-1])
    ax.set_xlabel("UT")
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax.set_ylabel("Altitude [deg]")
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo("UTC")))
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=ZoneInfo(obs.timezone)))
    ax_top.set_xlabel(f"Local time ({obs.timezone})")

    alt_ticks = np.array([20, 30, 40, 50, 60, 70, 80])
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(alt_ticks)
    ax2.set_yticklabels([f"{airmass_from_alt_deg(a):.2f}" for a in alt_ticks])
    ax2.set_ylabel("Airmass")

    ax.plot(times_ut, moon_alt, color="black", lw=3, ls=":", alpha=0.8)
    if len(times_ut) > 1:
        ax.text(times_ut[min(len(times_ut)-1, max(1, len(times_ut)//10))], float(np.nanmax(moon_alt)) if np.isfinite(np.nanmax(moon_alt)) else 5, "Moon", fontsize=9, color="black")

    if schedule_df is None or schedule_df.empty:
        ax.text(0.5, 0.5, "No observations scheduled for this night", ha="center", va="center", transform=ax.transAxes)
    else:
        sdf = schedule_df.copy().sort_values("start_local")
        cmap = plt.get_cmap("tab20")
        notes = []
        for i, row in enumerate(sdf.itertuples(index=False)):
            color = cmap(i % 20)
            coord = SkyCoord(float(row.ra_deg) * u.deg, float(row.dec_deg) * u.deg, frame="icrs")
            alt_deg = coord.transform_to(frame).alt.deg
            s0 = pd.to_datetime(row.start_local).to_pydatetime().astimezone(ZoneInfo("UTC"))
            s1 = pd.to_datetime(row.end_local).to_pydatetime().astimezone(ZoneInfo("UTC"))
            ax.axvspan(s0, s1, color=color, alpha=0.2)
            ax.plot(times_ut, alt_deg, color=color, alpha=0.4, lw=1)
            mask = np.array([(tt >= s0) and (tt <= s1) for tt in times_ut])
            if np.any(mask):
                ax.plot(np.array(times_ut)[mask], np.array(alt_deg)[mask], color=color, alpha=1.0, lw=2)
            hex_color = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
            notes.append((hex_color, f"{row.ob_name} ({int(round(row.exp_time_s))} s)"))

        x0, y0, dx, per_row = 0.02, 0.02, 0.30, 3
        for j, (hex_color, text) in enumerate(notes):
            rowi, coli = divmod(j, per_row)
            fig.text(x0 + coli * dx, y0 - rowi * 0.03, u"■", color=hex_color, fontsize=12, va="bottom")
            fig.text(x0 + coli * dx + 0.015, y0 - rowi * 0.03, text, fontsize=8, va="bottom")

    ax.set_title(f"Night visibility and allocated OBs — {night_date.isoformat()}")
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def altitude_airmass_plotly_figure(obs: ObservatoryConfig, night_date: date, bounds: Dict[str, datetime], schedule_df: pd.DataFrame, grid_minutes: int = 5):
    import plotly.graph_objects as go

    start = bounds.get("evening_astronomical")
    end = bounds.get("morning_astronomical")
    eve_civil = bounds.get("evening_civil")
    eve_naut = bounds.get("evening_nautical")
    eve_astro = bounds.get("evening_astronomical")
    morn_astro = bounds.get("morning_astronomical")
    morn_naut = bounds.get("morning_nautical")
    morn_civil = bounds.get("morning_civil")
    fig = go.Figure()
    if start is None or end is None:
        fig.update_layout(title="Night visibility and allocated OBs")
        return fig

    times_local = []
    t = eve_civil if eve_civil is not None else start
    tend = morn_civil if morn_civil is not None else end
    while t <= tend:
        times_local.append(t)
        t += timedelta(minutes=grid_minutes)
    atimes = Time(times_local)
    frame = AltAz(obstime=atimes, location=obs.location())
    times_ut = [tt.astimezone(ZoneInfo("UTC")) for tt in times_local]
    moon_alt = get_body("moon", atimes).transform_to(frame).alt.deg

    shapes, annotations = [], []
    def add_twilight(x0, x1, color, alpha, label, yloc, tcolor="black"):
        if x0 and x1:
            x0u, x1u = x0.astimezone(ZoneInfo("UTC")), x1.astimezone(ZoneInfo("UTC"))
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=x0u, x1=x1u, y0=0, y1=1, fillcolor=color, opacity=alpha, line_width=0, layer="below"))
            annotations.append(dict(x=x0u + (x1u - x0u) / 2, y=yloc, xref="x", yref="y", text=label, showarrow=False, font=dict(size=11, color=tcolor)))

    add_twilight(eve_civil, eve_naut, "lightgrey", 0.5, "Civil", 86)
    add_twilight(eve_naut, eve_astro, "grey", 0.5, "Nautical", 82, "white")
    add_twilight(eve_astro, start, "black", 0.5, "Astronomical", 78, "white")
    add_twilight(end, morn_astro, "black", 0.5, "Astronomical", 78, "white")
    add_twilight(morn_astro, morn_naut, "grey", 0.5, "Nautical", 82, "white")
    add_twilight(morn_naut, morn_civil, "lightgrey", 0.5, "Civil", 86)

    fig.add_trace(go.Scatter(x=times_ut, y=moon_alt, mode="lines", line=dict(color="black", width=3, dash="dot"), name="Moon", hovertemplate="Moon<br>UT: %{x|%H:%M:%S}<br>Altitude: %{y:.2f} deg<extra></extra>"))
    if len(times_ut) > 0:
        annotations.append(dict(x=times_ut[min(len(times_ut)-1, max(1, len(times_ut)//10))], y=float(np.nanmax(moon_alt)) if np.isfinite(np.nanmax(moon_alt)) else 5, xref="x", yref="y", text="Moon", showarrow=False, font=dict(size=11, color="black")))

    if schedule_df is not None and not schedule_df.empty:
        sdf = schedule_df.copy().sort_values("start_local")
        colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        for i, row in enumerate(sdf.itertuples(index=False)):
            color = colors_list[i % len(colors_list)]
            coord = SkyCoord(float(row.ra_deg) * u.deg, float(row.dec_deg) * u.deg, frame="icrs")
            alt_deg = coord.transform_to(frame).alt.deg
            s0 = pd.to_datetime(row.start_local).to_pydatetime().astimezone(ZoneInfo("UTC"))
            s1 = pd.to_datetime(row.end_local).to_pydatetime().astimezone(ZoneInfo("UTC"))
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=s0, x1=s1, y0=0, y1=1, fillcolor=color, opacity=0.2, line_width=0, layer="below"))
            fig.add_trace(go.Scatter(x=times_ut, y=alt_deg, mode="lines", line=dict(color=color, width=1), opacity=0.4, name=f"{row.ob_name} ({int(round(row.exp_time_s))} s)", hovertemplate=f"OB: {row.ob_name}<br>UT: %{{x|%H:%M:%S}}<br>Altitude: %{{y:.2f}} deg<br>Airmass: %{{customdata:.2f}}<extra></extra>", customdata=np.array([airmass_from_alt_deg(a) for a in alt_deg])))
            mask = np.array([(tt >= s0) and (tt <= s1) for tt in times_ut])
            if np.any(mask):
                fig.add_trace(go.Scatter(x=np.array(times_ut)[mask], y=np.array(alt_deg)[mask], mode="lines", line=dict(color=color, width=2), opacity=1.0, showlegend=False, hovertemplate=f"OB: {row.ob_name}<br>UT: %{{x|%H:%M:%S}}<br>Altitude: %{{y:.2f}} deg<br>Airmass: %{{customdata:.2f}}<extra></extra>", customdata=np.array([airmass_from_alt_deg(a) for a in np.array(alt_deg)[mask]])))

    alt_ticks = np.array([20, 30, 40, 50, 60, 70, 80])
    fig.update_layout(
        title=f"Night visibility and allocated OBs — {night_date.isoformat()}",
        xaxis=dict(title="UT", range=[times_ut[0], times_ut[-1]]),
        xaxis2=dict(title=f"Local time ({obs.timezone})", overlaying="x", side="top", range=[times_ut[0], times_ut[-1]], tickformat="%H:%M"),
        yaxis=dict(title="Altitude [deg]", range=[0, 90]),
        yaxis2=dict(title="Airmass", overlaying="y", side="right", tickmode="array", tickvals=alt_ticks.tolist(), ticktext=[f"{airmass_from_alt_deg(a):.2f}" for a in alt_ticks]),
        shapes=shapes,
        annotations=annotations,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
        margin=dict(l=60, r=80, t=60, b=140),
        height=800,
    )
    return fig


def dataframe_to_pdf_bytes(night_date: date, obs: ObservatoryConfig, bounds: Dict[str, datetime], schedule_df: pd.DataFrame, summary_df: Optional[pd.DataFrame] = None, run_config: Optional[Dict[str, str]] = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=0.3 * cm, rightMargin=0.3 * cm, topMargin=0.3 * cm, bottomMargin=0.3 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Nightly Observation Plan — {obs.name} — {night_date.isoformat()}", styles["Title"]), Spacer(1, 0.2 * cm)]
    if bounds:
        for line in [
            f"Evening civil twilight: {bounds['evening_civil']}",
            f"Evening nautical twilight: {bounds['evening_nautical']}",
            f"Evening astronomical twilight: {bounds['evening_astronomical']}",
            f"Morning astronomical twilight: {bounds['morning_astronomical']}",
            f"Morning nautical twilight: {bounds['morning_nautical']}",
            f"Morning civil twilight: {bounds['morning_civil']}",
        ]:
            story.append(Paragraph(line, styles["BodyText"]))
    story.append(Spacer(1, 0.2 * cm))
    if schedule_df.empty:
        story.append(Paragraph("No observations scheduled for this night.", styles["Heading2"]))
    else:
        show = schedule_df.copy().drop(columns=[c for c in ["block_id", "overhead_s"] if c in schedule_df.columns], errors="ignore")
        for col in ["start_local", "end_local"]:
            show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        show["airmass_start"] = show["airmass_start"].map(lambda x: f"{x:.2f}")
        show["airmass_mid"] = show["airmass_mid"].map(lambda x: f"{x:.2f}")
        show["moon_sep_deg"] = show["moon_sep_deg"].map(lambda x: f"{x:.1f}")
        cols = ["start_local", "end_local", "survey", "ob_name", "exp_time_s", "airmass_start", "airmass_mid", "moon_sep_deg", "sky_required", "sky_actual", "priority"]
        tbl = Table([cols] + show[cols].astype(str).values.tolist(), repeatRows=1, colWidths=[2.7*cm, 2.7*cm, 2.2*cm, 5.1*cm, 1.7*cm, 1.5*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.1*cm])
        tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])]))
        story.append(tbl)
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph("Night visibility plot", styles["Heading2"]))
        story.append(Image(io.BytesIO(altitude_airmass_plot_bytes(obs, night_date, bounds, schedule_df)), width=27 * cm, height=15 * cm))
    if summary_df is not None and not summary_df.empty:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Campaign completion by survey", styles["Heading2"]))
        s = summary_df.copy()
        s["required_total_hr"] = s["required_total_hr"].map(lambda x: f"{x:.2f}")
        s["observed_total_hr"] = s["observed_total_hr"].map(lambda x: f"{x:.2f}")
        s["completion_fraction"] = s["completion_fraction"].map(lambda x: f"{x:.3f}")
        stbl = Table([list(s.columns)] + s.astype(str).values.tolist(), repeatRows=1)
        stbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6.5)]))
        story.append(stbl)
    if run_config is not None:
        story.append(PageBreak())
        story.append(Paragraph("Run configuration / provenance", styles["Title"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(KeepTogether([config_table_for_pdf(run_config)]))
    doc.build(story)
    return buf.getvalue()


def campaign_pdf_bytes(obs: ObservatoryConfig, nightly_tables: Dict[date, pd.DataFrame], nightly_bounds: Dict[date, Dict[str, datetime]], summary_df: pd.DataFrame, progress_df: pd.DataFrame, run_config: Optional[Dict[str, str]] = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=0.3 * cm, rightMargin=0.3 * cm, topMargin=0.3 * cm, bottomMargin=0.3 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"Campaign Observation Plan — {obs.name}", styles["Title"]), Spacer(1, 0.2 * cm), Paragraph("Final survey completion table", styles["Heading2"])]
    if not summary_df.empty:
        s = summary_df.copy().sort_values("completion_fraction", ascending=False)
        s["required_total_hr"] = s["required_total_hr"].map(lambda x: f"{x:.2f}")
        s["observed_total_hr"] = s["observed_total_hr"].map(lambda x: f"{x:.2f}")
        s["completion_fraction"] = s["completion_fraction"].map(lambda x: f"{x:.3f}")
        stbl = Table([list(s.columns)] + s.astype(str).values.tolist(), repeatRows=1)
        stbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6.5)]))
        story.append(stbl)
    if progress_df is not None and not progress_df.empty:
        story += [Spacer(1, 0.2 * cm), Paragraph("Survey progress: completion fraction vs time", styles["Heading2"]), Image(io.BytesIO(make_progress_plot_bytes(progress_df)), width=27 * cm, height=14 * cm)]
    for night in sorted(nightly_tables.keys()):
        story.append(PageBreak())
        bounds = nightly_bounds.get(night, {})
        story.append(Paragraph(f"Nightly Observation Plan — {obs.name} — {night.isoformat()}", styles["Title"]))
        story.append(Spacer(1, 0.15 * cm))
        if bounds:
            for line in [
                f"Evening civil twilight: {bounds['evening_civil']}",
                f"Evening nautical twilight: {bounds['evening_nautical']}",
                f"Evening astronomical twilight: {bounds['evening_astronomical']}",
                f"Morning astronomical twilight: {bounds['morning_astronomical']}",
                f"Morning nautical twilight: {bounds['morning_nautical']}",
                f"Morning civil twilight: {bounds['morning_civil']}",
            ]:
                story.append(Paragraph(line, styles["BodyText"]))
        story.append(Spacer(1, 0.15 * cm))
        night_df = nightly_tables[night]
        if night_df.empty:
            story.append(Paragraph("No observations scheduled for this night.", styles["Heading2"]))
        else:
            show = night_df.copy().drop(columns=[c for c in ["block_id", "overhead_s"] if c in night_df.columns], errors="ignore")
            for col in ["start_local", "end_local"]:
                show[col] = pd.to_datetime(show[col]).dt.strftime("%Y-%m-%d %H:%M:%S")
            show["airmass_start"] = show["airmass_start"].map(lambda x: f"{x:.2f}")
            show["airmass_mid"] = show["airmass_mid"].map(lambda x: f"{x:.2f}")
            show["moon_sep_deg"] = show["moon_sep_deg"].map(lambda x: f"{x:.1f}")
            cols = ["start_local", "end_local", "survey", "ob_name", "exp_time_s", "airmass_start", "airmass_mid", "moon_sep_deg", "sky_required", "sky_actual", "priority"]
            tbl = Table([cols] + show[cols].astype(str).values.tolist(), repeatRows=1, colWidths=[2.7*cm, 2.7*cm, 2.2*cm, 5.1*cm, 1.7*cm, 1.5*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.1*cm])
            tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])]))
            story.append(tbl)
            story.append(Spacer(1, 0.15 * cm))
            story.append(Paragraph("Night visibility plot", styles["Heading2"]))
            story.append(Image(io.BytesIO(altitude_airmass_plot_bytes(obs, night, bounds, night_df)), width=27 * cm, height=15 * cm))
    if run_config is not None:
        story.append(PageBreak())
        story.append(Paragraph("Run configuration / provenance", styles["Title"]))
        story.append(Spacer(1, 0.2 * cm))
        story.append(KeepTogether([config_table_for_pdf(run_config)]))
    doc.build(story)
    return buf.getvalue()


def risk_pdf_bytes(obs: ObservatoryConfig, risk_df: pd.DataFrame, start_date: date, end_date: date) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=0.3 * cm, rightMargin=0.3 * cm, topMargin=0.3 * cm, bottomMargin=0.3 * cm)
    styles = getSampleStyleSheet()
    story = [Paragraph(f"OB Risk Diagnostics — {obs.name} — {start_date.isoformat()} to {end_date.isoformat()}", styles["Title"]), Spacer(1, 0.2 * cm)]
    if risk_df is None or risk_df.empty:
        story.append(Paragraph("No risk diagnostics available.", styles["Heading2"]))
    else:
        show = risk_df.copy()
        show["ra_deg"] = pd.to_numeric(show["ra_deg"], errors="coerce").map(lambda x: f"{x:.4f}")
        show["dec_deg"] = pd.to_numeric(show["dec_deg"], errors="coerce").map(lambda x: f"{x:.4f}")
        show["risk_quotient"] = pd.to_numeric(show["risk_quotient"], errors="coerce").map(lambda x: f"{x:.4f}")
        show["date_beyond_unobservable"] = pd.to_datetime(show["date_beyond_unobservable"], errors="coerce").dt.strftime("%Y-%m-%d")
        cols = ["risk_rank", "survey", "ob_name", "ra_deg", "dec_deg", "risk_quotient", "observable_nights", "date_beyond_unobservable"]
        tbl = Table([cols] + show[cols].astype(str).values.tolist(), repeatRows=1, colWidths=[1.2*cm, 4.0*cm, 5.2*cm, 1.8*cm, 1.8*cm, 2.0*cm, 2.6*cm, 3.6*cm])
        tbl.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7f0000")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 6.5), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey])]))
        story.append(tbl)
    doc.build(story)
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Dummy survey generation
# -----------------------------------------------------------------------------
def generate_realistic_survey_csv(survey_name: str, nrows: int = 10, seed: int = 42, observatory_lat_deg: float = -24.6667) -> bytes:
    rng = np.random.default_rng(seed)
    dec_core = rng.normal(loc=observatory_lat_deg, scale=18.0, size=nrows)
    dec_tail = rng.uniform(-70, 25, size=nrows)
    dec_deg = np.where(rng.random(nrows) < 0.25, dec_tail, dec_core)
    dec_deg = np.clip(dec_deg, -75, 30)
    n_clusters = int(rng.integers(2, 5))
    cluster_centers = rng.uniform(0, 360, size=n_clusters)
    cluster_ids = rng.integers(0, n_clusters, size=nrows)
    ra_deg = np.array([(rng.normal(cluster_centers[cid], 8.0) % 360.0) for cid in cluster_ids])
    sky_class = rng.choice(["dark", "grey", "bright"], size=nrows, p=[0.40, 0.35, 0.25])

    airmass_max, exp_time_s, min_moon_sep_angle, nexp, priority = [], [], [], [], []
    for sc in sky_class:
        if sc == "dark":
            airmass_max.append(rng.choice([1.4, 1.5, 1.6, 1.8], p=[0.15, 0.35, 0.30, 0.20]))
            exp_time_s.append(rng.choice([1200, 1800, 2400, 3000], p=[0.15, 0.35, 0.35, 0.15]))
            min_moon_sep_angle.append(rng.choice([40, 50, 60, 75], p=[0.15, 0.35, 0.35, 0.15]))
            nexp.append(rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30]))
            priority.append(rng.choice([1, 2, 3], p=[0.30, 0.50, 0.20]))
        elif sc == "grey":
            airmass_max.append(rng.choice([1.5, 1.8, 2.0], p=[0.30, 0.45, 0.25]))
            exp_time_s.append(rng.choice([900, 1200, 1800, 2400], p=[0.20, 0.35, 0.30, 0.15]))
            min_moon_sep_angle.append(rng.choice([25, 30, 40, 50], p=[0.20, 0.35, 0.30, 0.15]))
            nexp.append(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
            priority.append(rng.choice([1, 2, 3], p=[0.20, 0.55, 0.25]))
        else:
            airmass_max.append(rng.choice([1.8, 2.0, 2.2], p=[0.25, 0.50, 0.25]))
            exp_time_s.append(rng.choice([600, 900, 1200, 1800], p=[0.20, 0.35, 0.30, 0.15]))
            min_moon_sep_angle.append(rng.choice([15, 20, 25, 30], p=[0.20, 0.35, 0.30, 0.15]))
            nexp.append(rng.choice([1, 2], p=[0.60, 0.40]))
            priority.append(rng.choice([1, 2, 3], p=[0.15, 0.55, 0.30]))

    df = pd.DataFrame(
        {
            "survey": [survey_name] * nrows,
            "ob_name": [f"{survey_name}_OB_{i+1:03d}" for i in range(nrows)],
            "ra_deg": np.round(ra_deg, 6),
            "dec_deg": np.round(dec_deg, 6),
            "airmass_min": np.full(nrows, 1.0),
            "airmass_max": np.array(airmass_max),
            "exp_time_s": np.array(exp_time_s),
            "priority": np.array(priority),
            "sky_class": sky_class,
            "nexp": np.array(nexp),
            "min_moon_sep_angle": np.array(min_moon_sep_angle),
            "already_completed_obs": rng.choice([0, 0, 0, 1, 2], size=nrows, p=[0.55, 0.20, 0.10, 0.10, 0.05]),
        }
    ).sort_values(["ra_deg", "dec_deg"]).reset_index(drop=True)
    return df.to_csv(index=False).encode("utf-8")


def generate_many_realistic_surveys(ns: int = 50, nrows: int = 10, observatory_lat_deg: float = -24.6667) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_rows = []
        for i in range(ns):
            survey_name = f"Survey{i+1:02d}"
            data = generate_realistic_survey_csv(survey_name, nrows=nrows, seed=500 + i, observatory_lat_deg=observatory_lat_deg)
            zf.writestr(f"{survey_name}.csv", data)
            tmp = pd.read_csv(io.BytesIO(data))
            manifest_rows.append(
                {
                    "survey": survey_name,
                    "n_targets": len(tmp),
                    "ra_min": tmp["ra_deg"].min(),
                    "ra_max": tmp["ra_deg"].max(),
                    "dec_min": tmp["dec_deg"].min(),
                    "dec_max": tmp["dec_deg"].max(),
                }
            )
        zf.writestr("survey_manifest.csv", pd.DataFrame(manifest_rows).to_csv(index=False))
    return buf.getvalue()


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Observation Planner", layout="wide")
    st.title("4MOST Observation Block Planner")
    st.caption("OB scheduler with nightly plans, includes fairness and urgency options, and risk diagnostics (v1.5)")
    st.markdown("<hr style='border:4px solid #1f77b4'>", unsafe_allow_html=True)
    compute_now = False
    run_logs: List[str] = []
    timing_stats: Dict[str, float] = {}

    with st.sidebar:
        st.header("Observatory")
        obs_name = st.text_input("Observatory name", DEFAULT_OBSERVATORY["name"])
        lat = st.number_input("Latitude [deg]", value=float(DEFAULT_OBSERVATORY["latitude_deg"]), format="%.6f")
        lon = st.number_input("Longitude [deg]", value=float(DEFAULT_OBSERVATORY["longitude_deg"]), format="%.6f")
        elev = st.number_input("Elevation [m]", value=float(DEFAULT_OBSERVATORY["elevation_m"]), format="%.1f")
        tz = st.text_input("Timezone", DEFAULT_OBSERVATORY["timezone"])
        temp_c = st.number_input("Temperature [C]", value=float(DEFAULT_OBSERVATORY["temperature_c"]), format="%.1f")
        pressure_hpa = st.number_input("Pressure [hPa]", value=float(DEFAULT_OBSERVATORY["pressure_hpa"]), format="%.1f")
        humidity = st.number_input("Relative humidity [0-1]", value=float(DEFAULT_OBSERVATORY["relative_humidity"]), format="%.2f")

        st.header("Planning mode")
        planning_mode = st.radio("Choose planning scope", ["Plan for a night", "Plan for a period"], index=1)
        start_date = st.date_input("Start date", value=date.today())
        end_date = start_date if planning_mode == "Plan for a night" else st.date_input("End date", value=date.today() + timedelta(days=7))
        overhead_s = st.number_input("Global overhead per exposure [s]", min_value=0, value=DEFAULT_OVERHEAD_S, step=10)
        grid_minutes = st.number_input("Scheduler grid [minutes]", min_value=1, value=DEFAULT_GRID_MIN, step=1)
        risk_grid_minutes = st.number_input("Risk diagnostics / urgency grid [minutes]", min_value=5, value=180, step=5)
        include_risk_diagnostics = st.checkbox("Include OB risk diagnostics in planner run (only enable if planning for a period)", value=True)

        st.header("Sky classification")
        dark_illum_max = st.slider("Dark if moon illumination ≤", 0.0, 1.0, float(DEFAULT_MOON_ILLUM_DARK_MAX), 0.01)
        grey_illum_max = st.slider("Grey if moon illumination ≤", 0.0, 1.0, float(DEFAULT_MOON_ILLUM_GREY_MAX), 0.01)
        bright_alt_min_deg = st.slider("Moon altitude threshold [deg]", -18.0, 20.0, float(DEFAULT_MOON_ALT_BRIGHT_MIN_DEG), 0.5)

        st.header("Scheduling controls")
        fairness_enabled = st.checkbox("Enable fairness", value=True)
        fairness_weight = st.slider("Fairness weight", 0.0, 10.0, 5.0, 0.5, disabled=not fairness_enabled)
        priority_enabled = st.checkbox("Enable priority", value=True)
        priority_weight = st.slider("Priority weight", 0.0, 10.0, 1.0, 0.5, disabled=not priority_enabled)
        visibility_enabled = st.checkbox("Enable visibility", value=True)
        visibility_weight = st.slider("Visibility weight", 0.0, 10.0, 1.0, 0.5, disabled=not visibility_enabled)
        urgency_enabled = st.checkbox("Enable urgency (remaining observability)", value=True)
        urgency_weight = st.slider("Urgency weight", 0.0, 10.0, 4.0, 0.5, disabled=not urgency_enabled)
        run = st.button("Run planner", type="primary")

    obs = ObservatoryConfig(obs_name, lat, lon, elev, tz, temp_c, pressure_hpa, humidity)
    run_config = None

    st.subheader("Input schema")
    st.code(
        "survey,ob_name,ra_deg,dec_deg,airmass_min,airmass_max,exp_time_s,priority,sky_class,nexp,min_moon_sep_angle,already_completed_obs",
        language="text",
    )

    section_title("Generate realistic surveys input (for testing)", "#eefaf2")
    c1, c2 = st.columns(2)
    with c1:
        n_surveys = st.number_input("Number of surveys (N)", min_value=1, max_value=500, value=50, step=1)
    with c2:
        m_entries = st.number_input("Entries per survey (M)", min_value=1, max_value=1000, value=10, step=1)
    if st.button("Generate realistic survey files"):
        st.session_state["realistic_dummy_zip"] = generate_many_realistic_surveys(int(n_surveys), int(m_entries), observatory_lat_deg=lat)
    if "realistic_dummy_zip" in st.session_state:
        st.download_button(
            "Download generated realistic surveys",
            data=st.session_state["realistic_dummy_zip"],
            file_name=f"realistic_surveys_{int(n_surveys)}x{int(m_entries)}.zip",
            mime="application/zip",
        )

    uploaded = st.file_uploader("Upload one or more survey CSV files", type=["csv"], accept_multiple_files=True)
    raw_df = pd.DataFrame()
    if uploaded:
        try:
            with timed("parse_ob_files", run_logs, timing_stats):
                raw_df = parse_ob_files(uploaded, overhead_s=overhead_s, logs=run_logs)
            section_title("Merged survey input (Editable Table !)", "#f7f9fc")
            raw_df = pd.DataFrame(st.data_editor(raw_df, use_container_width=True, num_rows="dynamic", key="merged_survey_editor"))
            st.markdown("<hr style='border:4px solid #1f77b4'>", unsafe_allow_html=True)
            section_title("Best OBs at a given time", "#eefaf2")
            cb1, cb2 = st.columns([2, 1])
            with cb1:
                best_ob_ut = st.text_input("Enter UT time (HH:MM or HH:MM:SS)", value="00:00:00")
            with cb2:
                st.write("")
                compute_now = st.button("Compute", use_container_width=True)
        except Exception as e:
            st.error(f"Failed to parse uploaded files: {e}")
            return
    if compute_now and not raw_df.empty:
        try:
            with timed("best_obs_at_given_ut", run_logs, timing_stats):
                best_df = best_obs_at_given_ut(obs, raw_df, best_ob_ut, start_date, float(dark_illum_max), float(grey_illum_max), float(bright_alt_min_deg))
            st.session_state["best_ob_results"] = {"best_df": best_df, "ut_time": best_ob_ut, "ref_date": start_date}
        except Exception as e:
            st.exception(e)

    if run:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
            return
        if raw_df.empty:
            st.error("Please upload at least one valid survey CSV file.")
            return
        try:
            total_t0 = time.perf_counter()
            log_event("Planner run started", run_logs)

            with timed("expand_blocks", run_logs, timing_stats):
                blocks, survey_required_df = expand_blocks(raw_df, logs=run_logs)

            with timed("schedule_campaign", run_logs, timing_stats):
                nightly_tables, nightly_bounds, summary_df, progress_df, urgency_nights, observability_matrix = schedule_campaign(
                    obs=obs,
                    start_date=start_date,
                    end_date=end_date,
                    blocks=blocks,
                    survey_required_df=survey_required_df,
                    grid_minutes=int(grid_minutes),
                    urgency_grid_minutes=int(risk_grid_minutes),
                    fairness_weight=float(fairness_weight) if fairness_enabled else 0.0,
                    priority_weight=float(priority_weight) if priority_enabled else 0.0,
                    visibility_weight=float(visibility_weight) if visibility_enabled else 0.0,
                    urgency_weight=float(urgency_weight) if urgency_enabled else 0.0,
                    dark_illum_max=float(dark_illum_max),
                    grey_illum_max=float(grey_illum_max),
                    bright_alt_min_deg=float(bright_alt_min_deg),
                    logs=run_logs,
                    stats=timing_stats,
                )

            risk_df, risk_pdf = pd.DataFrame(), b""
            if include_risk_diagnostics:
                with timed("compute_block_risk_table", run_logs, timing_stats):
                    risk_df = compute_block_risk_table(blocks, start_date, urgency_nights, observability_matrix, run_logs)
                with timed("risk_pdf_bytes", run_logs, timing_stats):
                    risk_pdf = risk_pdf_bytes(obs, risk_df, start_date, end_date)

            now_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"QMOST_OBScheduler__{start_date.strftime('%Y%m%d')}to{end_date.strftime('%Y%m%d')}_{now_stamp}"
            code_filename = os.path.basename(__file__)
            run_config = build_run_config(
                obs=obs,
                planning_mode=planning_mode,
                start_date=start_date,
                end_date=end_date,
                overhead_s=float(overhead_s),
                grid_minutes=int(grid_minutes),
                urgency_grid_minutes=int(risk_grid_minutes),
                include_risk_diagnostics=bool(include_risk_diagnostics),
                dark_illum_max=float(dark_illum_max),
                grey_illum_max=float(grey_illum_max),
                bright_alt_min_deg=float(bright_alt_min_deg),
                fairness_enabled=bool(fairness_enabled),
                fairness_weight=float(fairness_weight),
                priority_enabled=bool(priority_enabled),
                priority_weight=float(priority_weight),
                visibility_enabled=bool(visibility_enabled),
                visibility_weight=float(visibility_weight),
                urgency_enabled=bool(urgency_enabled),
                urgency_weight=float(urgency_weight),
                output_name=base_name,
                code_filename=code_filename,
            )

            with timed("campaign_pdf_bytes", run_logs, timing_stats):
                campaign_pdf = campaign_pdf_bytes(obs, nightly_tables, nightly_bounds, summary_df, progress_df, run_config=run_config)

            with timed("zip_outputs", run_logs, timing_stats):
                zip_buffer = io.BytesIO()
                all_schedule_rows = []
                with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("survey_completion_summary.csv", summary_df.to_csv(index=False))
                    zf.writestr("survey_progress_history.csv", progress_df.to_csv(index=False))
                    for night, df_night in nightly_tables.items():
                        all_schedule_rows.append(df_night.assign(night=str(night)))
                        zf.writestr(f"night_{night.isoformat()}_schedule.csv", df_night.to_csv(index=False))
                    master = pd.concat(all_schedule_rows, ignore_index=True) if all_schedule_rows else pd.DataFrame()
                    zf.writestr("master_schedule.csv", master.to_csv(index=False))
                    zf.writestr("campaign_plan.pdf", campaign_pdf)
                    if include_risk_diagnostics and not risk_df.empty:
                        zf.writestr("ob_risk_diagnostics.csv", risk_df.to_csv(index=False))
                        zf.writestr("ob_risk_diagnostics.pdf", risk_pdf)

            total_runtime = time.perf_counter() - total_t0
            timing_stats["total_runtime"] = total_runtime
            planner_stats = {
                "total_runtime_s": total_runtime,
                "n_surveys": int(len(survey_required_df)),
                "n_schedulable_obs": int(len(blocks)),
                "n_dates": int(len(list(daterange(start_date, end_date)))),
                "scheduler_grid_min": int(grid_minutes),
                "urgency_grid_min": int(risk_grid_minutes),
                "timing": dict(sorted(timing_stats.items(), key=lambda x: x[1], reverse=True)),
            }
            log_event(f"Planner run finished in {total_runtime:.2f} s", run_logs)

            st.session_state["planner_results"] = {
                "nightly_tables": nightly_tables,
                "nightly_bounds": nightly_bounds,
                "summary_df": summary_df,
                "progress_df": progress_df,
                "campaign_pdf": campaign_pdf,
                "zip_bytes": zip_buffer.getvalue(),
                "base_name": base_name,
                "planning_mode": planning_mode,
                "risk_df": risk_df,
                "risk_pdf": risk_pdf,
                "logs": run_logs,
                "stats": planner_stats,
                "run_config": run_config,
            }
        except Exception as e:
            st.exception(e)
            return

    if "planner_results" in st.session_state:
        results = st.session_state["planner_results"]
        nightly_tables = results["nightly_tables"]
        nightly_bounds = results["nightly_bounds"]
        summary_df = results["summary_df"]
        progress_df = results["progress_df"]
        campaign_pdf = results["campaign_pdf"]
        base_name = results["base_name"]
        retained_mode = results.get("planning_mode", "Plan for a period")
        risk_df = results.get("risk_df", pd.DataFrame())
        risk_pdf = results.get("risk_pdf", b"")
        logs = results.get("logs", [])
        stats = results.get("stats", {})
        run_config = results.get("run_config", None)

        if "best_ob_results" in st.session_state:
            best_res = st.session_state["best_ob_results"]
            best_df = best_res.get("best_df", pd.DataFrame())
            section_title("Five best OBs observable now", "#eefaf2")
            st.subheader(f"Five best OBs observable at UT {best_res.get('ut_time', '')} on {best_res.get('ref_date', start_date)}")
            if best_df is None or best_df.empty:
                st.info("No OB satisfies the constraints at the requested UT.")
            else:
                st.dataframe(best_df, use_container_width=True)

        st.markdown("<hr style='border:4px solid #1f77b4'>", unsafe_allow_html=True)
        section_title("Survey completion and progress", "#eef4ff")
        if retained_mode != "Plan for a night":
            st.subheader("Survey completion summary")
            st.dataframe(summary_df, use_container_width=True)
            st.subheader("Survey progress")
            try:
                import plotly.graph_objects as go
                fig_prog = go.Figure()
                pdf_prog = progress_df.copy()
                pdf_prog["night"] = pd.to_datetime(pdf_prog["night"])
                for survey, grp in pdf_prog.groupby("survey"):
                    grp = grp.sort_values("night")
                    fig_prog.add_trace(go.Scatter(x=grp["night"], y=grp["completion_fraction"], mode="lines+markers", name=str(survey), hovertemplate="Survey: %{fullData.name}<br>Date: %{x|%Y-%m-%d}<br>Completed fraction: %{y:.3f}<extra></extra>"))
                fig_prog.update_layout(xaxis_title="Date", yaxis_title="Completion fraction", yaxis_range=[0, 1.05], hovermode="closest", height=550)
                st.plotly_chart(fig_prog, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not render Plotly progress chart: {e}")

        section_title("Download files", "#fff3e8")
        st.download_button("Download all nightly PDFs + CSVs (ZIP)", data=results["zip_bytes"], file_name=f"{base_name}.zip", mime="application/zip")
        st.download_button("Download appended campaign PDF", data=campaign_pdf, file_name=f"{base_name}.pdf", mime="application/pdf")
        if risk_df is not None and not risk_df.empty:
            st.download_button("Download OB risk diagnostics PDF", data=risk_pdf, file_name=f"{base_name}__risk_diagnostics.pdf", mime="application/pdf")

        section_title("Next night in plan and visibility", "#f3efff")
        selected_night = list(nightly_tables.keys())[0]
        st.subheader("Night plan" if retained_mode == "Plan for a night" else "Next night in plan")
        bounds = nightly_bounds.get(selected_night, {})
        if bounds:
            twi_df = pd.DataFrame(
                {
                    "Twilight": ["Civil", "Nautical", "Astronomical"],
                    "Evening": [
                        bounds["evening_civil"].strftime("%H:%M:%S") if bounds.get("evening_civil") else "",
                        bounds["evening_nautical"].strftime("%H:%M:%S") if bounds.get("evening_nautical") else "",
                        bounds["evening_astronomical"].strftime("%H:%M:%S") if bounds.get("evening_astronomical") else "",
                    ],
                    "Morning": [
                        bounds["morning_civil"].strftime("%H:%M:%S") if bounds.get("morning_civil") else "",
                        bounds["morning_nautical"].strftime("%H:%M:%S") if bounds.get("morning_nautical") else "",
                        bounds["morning_astronomical"].strftime("%H:%M:%S") if bounds.get("morning_astronomical") else "",
                    ],
                }
            )
            st.table(twi_df)
        night_display_df = nightly_tables[selected_night].drop(columns=[c for c in ["block_id", "overhead_s"] if c in nightly_tables[selected_night].columns], errors="ignore")
        st.dataframe(night_display_df, use_container_width=True)
        try:
            st.plotly_chart(altitude_airmass_plotly_figure(obs, selected_night, bounds, nightly_tables[selected_night]), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render nightly visibility plot: {e}")
        pdf_preview = dataframe_to_pdf_bytes(selected_night, obs, bounds, nightly_tables[selected_night], summary_df, run_config=run_config)
        st.download_button(f"Download PDF for {selected_night.isoformat()}", data=pdf_preview, file_name=f"{base_name}__night_{selected_night.isoformat()}_plan.pdf", mime="application/pdf")

        if risk_df is not None and not risk_df.empty:
            section_title("OB risk diagnostics (only run if plannig for a period)", "#fdeeee")
            st.dataframe(risk_df[["risk_rank", "survey", "ob_name", "ra_deg", "dec_deg", "risk_quotient", "date_beyond_unobservable"]], use_container_width=True)

        section_title("Run logs and timing", "#eef8ff")
        if logs:
            st.text_area("Execution log", value="\n".join(logs), height=260)
        if stats:
            summary_rows = [
                {"Metric": "Total runtime [s]", "Value": f"{stats.get('total_runtime_s', 0):.2f}"},
                {"Metric": "Number of surveys", "Value": stats.get("n_surveys", 0)},
                {"Metric": "Number of schedulable OBs", "Value": stats.get("n_schedulable_obs", 0)},
                {"Metric": "Number of dates", "Value": stats.get("n_dates", 0)},
                {"Metric": "Scheduler grid [min]", "Value": stats.get("scheduler_grid_min", 0)},
                {"Metric": "Urgency/Risk grid [min]", "Value": stats.get("urgency_grid_min", 0)},
            ]
            st.table(pd.DataFrame(summary_rows))
            timing = stats.get("timing", {})
            if timing:
                st.subheader("Code Profiling Analysis")
                st.table(pd.DataFrame([{"Step": k, "Seconds": f"{v:.2f}"} for k, v in timing.items()]))

        if st.button("Clear retained results"):
            del st.session_state["planner_results"]
            st.rerun()
    st.markdown("<hr style='border:4px solid #1f77b4'>", unsafe_allow_html=True)
    with st.expander("Development notes"):
        st.markdown(
            """
            ## Author

            **Dr. Vivek M.**  
            Indian Institute of Astrophysics (IIA)  
            Bangalore, India  

            For questions, feedback,  the author can be contacted via email:

            vivek.m@iiap.res.in, getkeviv@gmail.com
            ## What the Software Does

            The **4MOST Observation Block Planner** is a scheduling tool designed to generate optimized observing plans for astronomical surveys and multi-program telescope operations. The software ingests observing requests containing target coordinates, exposure requirements, and observational constraints, and computes when each target can be observed based on astronomical visibility, airmass limits, sky brightness conditions, and Moon separation. It then applies a weighted scheduling model that balances **survey fairness (uniform completeness across surveys), scientific priority, target visibility, and observational urgency** to determine which observation should be scheduled at each time slot. The planner produces nightly observing schedules, campaign-level summaries, survey progress statistics, and risk diagnostics identifying targets that may become difficult to observe later in the campaign. This enables efficient and transparent allocation of telescope time across competing scientific programs.
            """
        )
        manual_pdf_path = "manual/Telescope_Observation_Planner_Full_Manual_v2.pdf"

    if os.path.exists(manual_pdf_path):
        with open(manual_pdf_path, "rb") as f:
            manual_pdf_bytes = f.read()

        st.markdown("### User Manual")
        st.download_button(
            "See User Manual here",
            data=manual_pdf_bytes,
            file_name="Telescope_Observation_Planner_Full_Manual_v2.pdf",
            mime="application/pdf",
        )
    else:
        st.info("User manual PDF not found.")

if __name__ == "__main__":
    main()

