"""Mapping tables from each dataset's own class names to our four classes.

Step 1 of the guide: "write a small mapping table from each dataset's class
names to your four classes". Names are normalised (upper case, letters and
digits only) before lookup, so "F/A-18", "F-18" and "FA18" all match.
Anything not listed is skipped, and data_prep prints what it skipped so you
can extend the tables.
"""
from __future__ import annotations

import re


def norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(name).upper())


# --------------------------------------------------------------------------
# FGVC-Aircraft, annotation_level="family" (70 families)
# --------------------------------------------------------------------------
_FGVC = {
    "passenger_jet": [
        "A300", "A310", "A320", "A330", "A340", "A380", "BAE 146", "BAE-125",
        "Boeing 707", "Boeing 717", "Boeing 727", "Boeing 737", "Boeing 747",
        "Boeing 757", "Boeing 767", "Boeing 777", "CRJ-200", "CRJ-700",
        "Cessna Citation", "Challenger 600", "DC-10", "DC-8", "DC-9",
        "E-170", "E-190", "E-195", "ERJ 135", "ERJ 145", "Embraer Legacy 600",
        "Falcon 2000", "Falcon 900", "Fokker 100", "Fokker 70",
        "Global Express", "Gulfstream", "Il-76", "L-1011", "MD-11", "MD-80",
        "MD-90", "Tu-134", "Tu-154", "Yak-42",
    ],
    "fighter_jet": [
        "Eurofighter Typhoon", "F-16", "F/A-18", "Hawk T1", "Tornado",
    ],
    "propeller": [
        "ATR-42", "ATR-72", "An-12", "Beechcraft 1900", "C-130", "C-47",
        "Cessna 172", "Cessna 208", "DC-3", "DC-6", "DH-82", "DHC-1", "DHC-6",
        "DR-400", "Dash 8", "Dornier 328", "EMB-120", "Fokker 50", "King Air",
        "Metroliner", "PA-28", "SR-20", "Saab 2000", "Saab 340", "Spitfire",
    ],
    # FGVC-Aircraft has no helicopters; they come from the military dataset.
}

# --------------------------------------------------------------------------
# Military Aircraft Detection Dataset (Kaggle). Class names vary a little
# between dataset versions, so the lists are generous.
# --------------------------------------------------------------------------
_MILITARY = {
    "fighter_jet": [
        "A10", "AV8B", "EF2000", "F4", "F14", "F15", "F16", "F18", "FA18",
        "F22", "F35", "F117", "J10", "J20", "J35", "JAS39", "JF17", "KAAN",
        "KF21", "MIG29", "MIG31", "MIRAGE2000", "RAFALE", "SU24", "SU25",
        "SU27", "SU30", "SU34", "SU57", "TORNADO", "YF23", "J36", "F111",
    ],
    "helicopter": [
        "AH64", "CH47", "CH53", "KA27", "KA52", "MI8", "MI24", "MI26", "MI28",
        "UH60", "Z10", "Z19", "AW101", "AW159", "NH90", "TIGER", "EC665",
        "MH60", "SH60", "AH1", "UH1",
    ],
    "propeller": [
        "C130", "A400M", "P3", "E2", "AN12", "AN22", "TU95", "C2", "AN70",
        "Y8", "Y9", "C27J", "C295", "CN235", "OV10", "EMB314", "PC7", "PC9",
        "T6",
    ],
    # Bombers, tankers, drones and tilt-rotors are deliberately left out:
    # they do not fit the four classes cleanly.
}

# --------------------------------------------------------------------------
# AeroSonicDB "subclass" column:
#   0 no aircraft, 1 piston-propeller, 2 turbine-propeller,
#   3 turbine-fan (jet), 4 rotorcraft (helicopter)
# The recordings are near a civil airport, so there are no fighters.
# --------------------------------------------------------------------------
AEROSONIC_SUBCLASS = {
    1: "propeller",
    2: "propeller",
    3: "passenger_jet",
    4: "helicopter",
}


def _invert(table):
    out = {}
    for cls, names in table.items():
        for n in names:
            out[norm(n)] = cls
    return out


FGVC_MAP = _invert(_FGVC)
MILITARY_MAP = _invert(_MILITARY)


def map_fgvc(family: str):
    return FGVC_MAP.get(norm(family))


def map_military(name: str):
    return MILITARY_MAP.get(norm(name))


def map_video_folder(name: str):
    """Video clips live in folders named after our classes (or close to it)."""
    n = norm(name)
    aliases = {
        "PASSENGERJET": "passenger_jet", "PASSENGER": "passenger_jet", "JET": "passenger_jet",
        "AIRLINER": "passenger_jet", "FIGHTERJET": "fighter_jet", "FIGHTER": "fighter_jet",
        "HELICOPTER": "helicopter", "HELI": "helicopter", "PROPELLER": "propeller",
        "PROP": "propeller", "PROPELLERAIRCRAFT": "propeller",
    }
    return aliases.get(n)
