from __future__ import annotations

import configparser
import os
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Settings:
    structure: Optional[str] = None
    mode: str = "auto"
    cutoff: Optional[float] = None
    hist_rmin: float = 1.8
    hist_rmax: float = 4.2
    surface_threshold: int = 12
    r_nbr: float = 3.6
    max_angle: float = 12.0
    tag: str = "NP"
    planarity: float = 0.02
    edge_tol: float = 0.10
    index_facets: str = "none"


def _coerce(name: str, val: str):
    if name in {"mode", "tag", "index_facets"}:
        return str(val)
    if name in {"structure"}:
        return str(val)
    if name in {"cutoff", "hist_rmin", "hist_rmax", "r_nbr", "max_angle", "planarity", "edge_tol"}:
        return float(val)
    if name in {"surface_threshold"}:
        return int(val)
    try:
        return float(val)
    except Exception:
        try:
            return int(val)
        except Exception:
            return val


def _read_first_existing(paths):
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def load_config(config_path=None, cli_args=None):
    cfg = {}
    if config_path:
        parser = configparser.ConfigParser()
        parser.read(config_path)
        for section in parser.sections():
            for key, val in parser.items(section):
                cfg[key.replace("-", "_")] = val

    if cli_args:
        for key, val in vars(cli_args).items():
            if val is not None:
                cfg[key] = val

    int_keys = ["surface_threshold"]
    float_keys = ["cutoff", "r_nbr", "max_angle", "planarity", "edge_tol"]
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])
    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])

    return cfg


def load_settings(cli_args) -> tuple[Settings, str | None]:
    """
    Load settings with precedence:
      CLI --config -> ./npf.count.in -> ./config.cn -> ~/.np_sites/npf.count.in -> ~/.np_sites/config.cn
    Then overlay with CLI flags that are not None.
    """
    cfg = Settings()

    home_count_cfg = os.path.expanduser("~/.np_sites/npf.count.in")
    home_legacy_cfg = os.path.expanduser("~/.np_sites/config.cn")
    chosen = _read_first_existing([
        getattr(cli_args, "config", None),
        "npf.count.in",
        "config.cn",
        home_count_cfg,
        home_legacy_cfg,
    ])

    if chosen:
        parser = configparser.ConfigParser()
        parser.read(chosen)

        section = None
        if parser.has_section("np_sites"):
            section = parser["np_sites"]
        elif parser.has_section("general"):
            section = parser["general"]

        if section:
            for key in section:
                if not hasattr(cfg, key):
                    continue
                val = section.get(key)
                if val is None:
                    continue
                try:
                    setattr(cfg, key, _coerce(key, val))
                except Exception:
                    pass

    for field in asdict(cfg).keys():
        if not hasattr(cli_args, field):
            continue
        v = getattr(cli_args, field)
        if v is not None:
            setattr(cfg, field, v)

    if cfg.mode == "fixed" and (cfg.cutoff is None or cfg.cutoff <= 0):
        raise ValueError("mode=fixed requires a positive --cutoff (or cutoff in npf.count.in).")

    valid_facets = {"none", "fcc", "hcp"}
    if not hasattr(cfg, "index_facets") or str(cfg.index_facets).lower() not in valid_facets:
        print(f"[config] Warning: invalid or missing index_facets='{getattr(cfg, 'index_facets', None)}'; defaulting to 'none'.")
        cfg.index_facets = "none"

    if cfg.surface_threshold < 1:
        raise ValueError("surface_threshold must be >= 1")

    return cfg, chosen

