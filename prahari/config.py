"""Loads config/settings.yaml and config/cameras.yaml.

Every tunable in the system comes from here. If you find a threshold hard-coded
in the Python, that is a bug - fix it by adding a key to settings.yaml.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

# Zones drawn in the dashboard land here rather than being written back over
# cameras.yaml, which is hand-written and full of comments worth keeping.
# Delete this file to go back to the shipped zones.
ZONES_OVERRIDE = CONFIG_DIR / "zones.local.yaml"


@dataclass
class CameraCfg:
    id: str
    name: str
    url: str
    rules: list[str] = field(default_factory=list)
    zones: dict[str, list[list[float]]] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class Settings:
    detect_fps: int
    img_width: int
    conf_threshold: float
    model: str
    imgsz: int
    device: str
    tracker: str
    store_open: str
    store_close: str
    demo_force_closed: bool
    thresholds: dict[str, float]
    cooldown_s: float
    rule_cooldown_s: dict[str, float]
    clip: dict[str, float]
    perf: dict[str, float]
    whatsapp: dict[str, Any]
    host: str
    port: int
    data_dir: Path
    raw: dict[str, Any] = field(default_factory=dict)

    def cooldown_for(self, rule: str) -> float:
        return float(self.rule_cooldown_s.get(rule, self.cooldown_s))

    def threshold(self, name: str) -> float:
        try:
            return float(self.thresholds[name])
        except KeyError as exc:
            raise KeyError(
                f"threshold '{name}' missing from config/settings.yaml"
            ) from exc

    def is_closed(self, ts: float) -> bool:
        """Is the store shut at this wall-clock time?"""
        if self.demo_force_closed:
            return True
        now = dt.datetime.fromtimestamp(ts).time()
        open_t = _parse_time(self.store_open)
        close_t = _parse_time(self.store_close)
        if open_t <= close_t:
            return not (open_t <= now < close_t)
        # Opening hours that straddle midnight.
        return close_t <= now < open_t

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def snapshots_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "prahari.db"


def _parse_time(value: str) -> dt.time:
    hh, mm = str(value).split(":")
    return dt.time(int(hh), int(mm))


def load_settings(path: Path | None = None) -> Settings:
    path = Path(path) if path else CONFIG_DIR / "settings.yaml"
    raw = yaml.safe_load(path.read_text()) or {}

    hours = raw.get("store_hours", {})
    data_dir = Path(raw.get("data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir

    s = Settings(
        detect_fps=int(raw.get("detect_fps", 5)),
        img_width=int(raw.get("img_width", 640)),
        conf_threshold=float(raw.get("conf_threshold", 0.4)),
        model=str(raw.get("model", "yolo11n.pt")),
        imgsz=int(raw.get("imgsz", 640)),
        device=str(raw.get("device", "auto")),
        tracker=str(raw.get("tracker", "bytetrack.yaml")),
        store_open=str(hours.get("open", "10:30")),
        store_close=str(hours.get("close", "20:30")),
        demo_force_closed=bool(raw.get("demo_force_closed", False)),
        thresholds=dict(raw.get("thresholds", {})),
        cooldown_s=float(raw.get("cooldown_s", 120)),
        rule_cooldown_s=dict(raw.get("rule_cooldown_s", {})),
        clip=dict(raw.get("clip", {"pre_s": 5, "post_s": 10})),
        perf=dict(raw.get("perf", {"slow_ms": 80, "slow_detect_fps": 3})),
        whatsapp=dict(raw.get("whatsapp", {})),
        host=str(raw.get("server", {}).get("host", "0.0.0.0")),
        port=int(raw.get("server", {}).get("port", 8000)),
        data_dir=data_dir,
        raw=raw,
    )
    s.clips_dir.mkdir(parents=True, exist_ok=True)
    s.snapshots_dir.mkdir(parents=True, exist_ok=True)
    return s


def load_zone_overrides() -> dict[str, dict]:
    if not ZONES_OVERRIDE.exists():
        return {}
    return (yaml.safe_load(ZONES_OVERRIDE.read_text()) or {}).get("zones", {})


def save_zone_override(cam_id: str, zones: dict[str, list]) -> None:
    """Persist zones drawn in the dashboard, leaving cameras.yaml untouched."""
    current = load_zone_overrides()
    current[cam_id] = zones
    ZONES_OVERRIDE.write_text(
        "# Written by the dashboard's Zones mode. Delete this file to go back\n"
        "# to the zones defined in cameras.yaml.\n"
        + yaml.safe_dump({"zones": current}, sort_keys=True, default_flow_style=None)
    )


def load_cameras(path: Path | None = None) -> list[CameraCfg]:
    path = Path(path) if path else CONFIG_DIR / "cameras.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    overrides = load_zone_overrides()
    cams: list[CameraCfg] = []
    for entry in raw.get("cameras", []):
        declared = overrides.get(str(entry["id"])) or entry.get("zones") or {}
        zones = {k: [list(p) for p in v] for k, v in declared.items()}
        cams.append(
            CameraCfg(
                id=str(entry["id"]),
                name=str(entry.get("name", entry["id"])),
                url=str(entry["url"]),
                rules=list(entry.get("rules", [])),
                zones=zones,  # mutated in place on save, so rules hot-reload
                enabled=bool(entry.get("enabled", True)),
            )
        )
    if not cams:
        raise ValueError(f"no cameras defined in {path}")
    return cams
