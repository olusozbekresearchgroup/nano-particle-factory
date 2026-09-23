import os
import configparser
from dataclasses import dataclass, asdict
from typing import Optional

# Single place for defaults (type-safe)
@dataclass
class Settings:
    structure: Optional[str] = None           # filled from CLI only
    mode: str = "auto"                        # fixed | auto | adaptive12
    cutoff: Optional[float] = None            # Å (required if mode=fixed)
    hist_rmin: float = 1.8
    hist_rmax: float = 4.2
    surface_threshold: int = 12
    r_nbr: float = 3.6
    max_angle: float = 12.0
    tag: str = "NP"
    planarity: float = 0.02
    edge_tol: float = 0.10
    index_facets: str = "none"                # none | fcc | hcp
    # IO names are fixed by code, but you can add toggles here if desired.

# ---- helpers ----
def _str2bool(x: str) -> bool:
    return x.strip().lower() in {"1","true","t","yes","y","on"}

def _coerce(name: str, val: str):
    # basic typed coercion by field name
    if name in {"mode","tag","index_facets"}:
        return str(val)
    if name in {"structure"}:
        return str(val)
    if name in {"cutoff","hist_rmin","hist_rmax","r_nbr","max_angle","planarity","edge_tol"}:
        return float(val)
    if name in {"surface_threshold"}:
        return int(val)
    # fallback: try float, then int, else raw
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
                cfg[key.replace('-', '_')] = val

    if cli_args:
        for key, val in vars(cli_args).items():
            if val is not None:
                cfg[key] = val

    # Convert types
    int_keys = ["surface_threshold"]
    float_keys = ["cutoff", "r_nbr", "max_angle", "planarity", "edge_tol"]
    for k in int_keys:
        if k in cfg: cfg[k] = int(cfg[k])
    for k in float_keys:
        if k in cfg: cfg[k] = float(cfg[k])

    return cfg

def load_settings(cli_args) -> Settings:
    """
    Load settings with precedence:
      CLI --config -> ./config.cn -> ~/.np_sites/config.cn
    Then overlay with CLI flags that are not None.
    """
    # 1) start with defaults
    cfg = Settings()

    # 2) locate config file
    home_cfg = os.path.expanduser("~/.np_sites/config.cn")
    chosen = _read_first_existing([getattr(cli_args, "config", None), "config.cn", home_cfg])

    if chosen:
        parser = configparser.ConfigParser()
        parser.read(chosen)

        # Accept values under [np_sites] and/or [general]
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
                    coerced = _coerce(key, val)
                    setattr(cfg, key, coerced)
                except Exception:
                    # Keep default on bad type
                    pass

    # 3) overlay CLI flags (only those explicitly provided)
    # NOTE: argparse fills with None for not-provided optionals
    for field in asdict(cfg).keys():
        if not hasattr(cli_args, field):
            continue
        v = getattr(cli_args, field)
        if v is not None:
            setattr(cfg, field, v)

    # 4) validation
    if cfg.mode == "fixed" and (cfg.cutoff is None or cfg.cutoff <= 0):
        raise ValueError("mode=fixed requires a positive --cutoff (or cutoff in config.cn).")

    # make index_facets forgiving — default to 'none' if invalid or missing
    valid_facets = {"none", "fcc", "hcp"}
    if not hasattr(cfg, "index_facets") or str(cfg.index_facets).lower() not in valid_facets:
        print(f"[config] Warning: invalid or missing index_facets='{getattr(cfg, 'index_facets', None)}'; defaulting to 'none'.")
        cfg.index_facets = "none"

    if cfg.surface_threshold < 1:
        raise ValueError("surface_threshold must be >= 1")


    return cfg, chosen

