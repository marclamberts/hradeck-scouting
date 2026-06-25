"""
build_lamberts_index.py
────────────────────────────────────────────────────────────────────
Lamberts Index — Full Player Intelligence Report (All Positions)
Covers: GK · CB · FB · DM · CM · W · FW

Core metrics computed:
  LI Score  – Lamberts Index Score (0–100) = SQS-based percentile per position
  LI         – Lamberts Index = LI Score Rank − Market Value Rank
              Positive = player performs above market price

Defensive-only sub-indexes (GK / CB / FB / DM):
  DIS  – Defensive Impact Score (weighted def output rank)
  ADI  – Aerial Dominance Index (aerial vol × quality, ranked)
  PADS – Pressure-Adjusted Defensive Score (PAdj metrics)
  PDS  – Positional Discipline Score (inverted foul-cost rank)

Usage:
  python build_lamberts_index.py
  python build_lamberts_index.py --leagues Czech Slovakia --min-minutes 500
  python build_lamberts_index.py --output data/Lamberts_Index_Report.xlsm
"""
from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import xlsxwriter
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ── Config ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
WYSCOUT_DIR = ROOT / "Wyscout Files"

SKIP_FILES = {
    "FCHK Model V3 - Loaded Leagues", "FCHK Model V3 - Model Input",
    "FCHK Model V3 - Player Scores", "FCHK Model V3 - Player Styles",
    "FCHK Model V3 - Recruitment Scores", "FCHK Model V3 - Smart Club Closeness",
    "FCHK Model V3 - Summary", "FCHK Model V3 Scores", "FCHK Scouting Report",
    "Leagues Overview", "Wyscout Anomaly Report", "Wyscout Full Scouting Report",
    "FCHK Model V3 Scores",
}

DEFAULT_MIN_MINUTES = 400
DEFAULT_MAX_AGE = 30
DEFAULT_BUDGET = 1_000_000

ALL_POSITIONS = {"GK", "CB", "FB", "DM", "CM", "W", "FW"}
DEFENSIVE_POSITIONS = {"GK", "CB", "FB", "DM"}

POS_MAP: dict[str, str] = {
    "CF": "FW", "SS": "FW",
    "LW": "W",  "RW": "W",  "LWF": "W",  "RWF": "W",  "WF": "W",
    "LAMF": "W", "RAMF": "W",
    "AMF": "CM",
    "CMF": "CM", "LCM": "CM", "RCM": "CM", "LCMF": "CM", "RCMF": "CM",
    "DMF": "DM", "LDM": "DM", "RDM": "DM", "LDMF": "DM", "RDMF": "DM",
    "LB": "FB",  "RB": "FB",  "LWB": "FB", "RWB": "FB",
    "CB": "CB",  "LCB": "CB", "RCB": "CB",
    "GK": "GK",
}

SQS_BLUEPRINTS: dict[str, list[tuple[str, float]]] = {
    "GK": [
        ("Save rate, %",              4.0),
        ("Prevented goals per 90",    3.0),
        ("Exits per 90",              2.0),
        ("Aerial duels per 90",       1.5),
        ("Accurate passes, %",        1.5),
        ("Accurate long passes, %",   1.0),
    ],
    "CB": [
        ("Successful defensive actions per 90", 3.0),
        ("Defensive duels won, %",              2.5),
        ("Aerial duels won, %",                 2.0),
        ("Interceptions per 90",                2.0),
        ("PAdj Interceptions",                  1.5),
        ("Shots blocked per 90",                1.0),
        ("Accurate passes, %",                  1.0),
        ("Progressive passes per 90",           1.0),
    ],
    "FB": [
        ("Crosses per 90",                      2.0),
        ("Accurate crosses, %",                 1.5),
        ("xA per 90",                           2.0),
        ("Assists per 90",                      1.5),
        ("Progressive runs per 90",             1.5),
        ("Dribbles per 90",                     1.0),
        ("Successful defensive actions per 90", 2.0),
        ("Defensive duels won, %",              2.0),
        ("Aerial duels won, %",                 1.0),
        ("Progressive passes per 90",           1.5),
    ],
    "DM": [
        ("Successful defensive actions per 90", 3.0),
        ("Defensive duels won, %",              2.5),
        ("Interceptions per 90",                2.5),
        ("PAdj Interceptions",                  2.0),
        ("Aerial duels won, %",                 1.5),
        ("Passes per 90",                       1.5),
        ("Accurate passes, %",                  1.5),
        ("Progressive passes per 90",           1.5),
    ],
    "CM": [
        ("Passes per 90",                       2.0),
        ("Accurate passes, %",                  2.0),
        ("Progressive passes per 90",           2.5),
        ("Key passes per 90",                   2.5),
        ("xA per 90",                           2.0),
        ("Assists per 90",                      1.5),
        ("Progressive runs per 90",             1.5),
        ("Successful defensive actions per 90", 1.5),
        ("Goals per 90",                        1.5),
    ],
    "W": [
        ("Goals per 90",                        2.5),
        ("xG per 90",                           2.0),
        ("Assists per 90",                      2.0),
        ("xA per 90",                           2.0),
        ("Dribbles per 90",                     2.0),
        ("Successful dribbles, %",              1.5),
        ("Progressive runs per 90",             2.0),
        ("Touches in box per 90",               2.0),
        ("Key passes per 90",                   1.5),
    ],
    "FW": [
        ("Goals per 90",                        4.0),
        ("xG per 90",                           3.0),
        ("Non-penalty goals per 90",            2.5),
        ("Shots per 90",                        1.5),
        ("Shots on target, %",                  1.5),
        ("Goal conversion, %",                  1.5),
        ("Touches in box per 90",               1.5),
        ("Aerial duels won, %",                 1.5),
        ("Dribbles per 90",                     1.0),
        ("xA per 90",                           1.0),
    ],
}

HRADEC_SQUAD: dict[str, float] = {
    "GK": 61.5,
    "CB": 33.3,
    "FB": 81.8,
    "DM": 36.8,
    "CM": 53.7,
    "W":  27.6,
    "FW": 38.3,
}

POS_LABELS = {
    "GK": "GK — GOALKEEPER",
    "CB": "CB — CENTRE-BACK",
    "FB": "FB — FULL-BACK",
    "DM": "DM — DEFENSIVE MID",
    "CM": "CM — CENTRAL MID",
    "W":  "W — WINGER",
    "FW": "FW — FORWARD",
}

# ── Colors ─────────────────────────────────────────────────────────────────────

C = {
    "navy":    "0D1B2A",
    "gold":    "C9A84C",
    "teal":    "0B6E4F",
    "elite":   "1A5276",
    "high":    "1E8449",
    "value":   "117A65",
    "fair":    "626567",
    "over":    "922B21",
    "white":   "FFFFFF",
    "light":   "EBF5FB",
    "ice":     "E8F8F5",
    "header":  "154360",
    "upgrade": "1A5276",
    "rota":    "7D6608",
    "depth":   "117A65",
    "li_dark": "0D1B2A",
    "li_pos":  "1E8449",
    "li_neg":  "922B21",
}

# ── VBA Macro (Filter Panel) ───────────────────────────────────────────────────

MODULE_NAME = "LambertsFilters"

VBA_SOURCE = f'''Attribute VB_Name = "{MODULE_NAME}"

Sub ApplyFilters()
    Dim fp As Worksheet
    Dim ws As Worksheet
    Dim pos As String, tier As String, status As String
    On Error Resume Next
    Set fp = ThisWorkbook.Sheets("Filter Panel")
    If fp Is Nothing Then
        MsgBox "Filter Panel sheet not found!", vbExclamation
        Exit Sub
    End If
    On Error GoTo 0
    pos    = Trim(CStr(fp.Range("B2").Value))
    tier   = Trim(CStr(fp.Range("B3").Value))
    status = Trim(CStr(fp.Range("B4").Value))
    For Each ws In ThisWorkbook.Worksheets
        Select Case ws.Name
            Case "Filter Panel", "README", "Lamberts Analysis", "Top 5", "Budget Planner", "Squad"
            Case Else
                If ws.AutoFilterMode Then ws.AutoFilterMode = False
                If ws.UsedRange.Rows.Count > 3 Then
                    ws.Rows(3).AutoFilter
                    If pos <> "" And pos <> "All" Then
                        ws.Rows(3).AutoFilter Field:=4, Criteria1:=pos
                    End If
                    If tier <> "" And tier <> "All" Then
                        ws.Rows(3).AutoFilter Field:=12, Criteria1:=tier
                    End If
                    If status <> "" And status <> "All" Then
                        ws.Rows(3).AutoFilter Field:=19, Criteria1:=status
                    End If
                End If
        End Select
    Next ws
    MsgBox "Filters applied to all sheets!", vbInformation
End Sub

Sub ClearAllFilters()
    Dim ws As Worksheet
    For Each ws In ThisWorkbook.Worksheets
        If ws.AutoFilterMode Then
            ws.AutoFilterMode = False
        End If
    Next ws
    MsgBox "All filters cleared!", vbInformation
End Sub
'''

# ── OLE2 / vbaProject.bin builder ─────────────────────────────────────────────

FREESECT   = 0xFFFFFFFF
ENDOFCHAIN = 0xFFFFFFFE
FATSECT    = 0xFFFFFFFD
NOSTREAM   = 0xFFFFFFFF


def _compress_ovba(data: bytes) -> bytes:
    """MS-OVBA uncompressed-chunk compression (simplest valid format)."""
    out = bytearray([0x01])
    i = 0
    while i < len(data):
        chunk = data[i:i + 4096]
        hdr = (len(chunk) - 1) & 0x0FFF  # bit15=0 → uncompressed
        out += struct.pack('<H', hdr)
        out += chunk
        i += 4096
    return bytes(out)


def _pad_to(data: bytes, size: int) -> bytes:
    if len(data) < size:
        return data + b'\x00' * (size - len(data))
    return data


def _pad512(data: bytes) -> bytes:
    r = len(data) % 512
    return data + b'\x00' * (512 - r) if r else data


def _pad4096(data: bytes) -> bytes:
    """Pad to ≥4096 bytes so streams bypass OLE2 mini-stream."""
    return _pad512(_pad_to(data, 4096))


def _dir_entry(name, obj_type, color, left, right, child, clsid_hex, start, size):
    enc = name.encode('utf-16-le') if name else b''
    nlen = len(enc) + 2 if name else 0
    e = bytearray(128)
    e[0:len(enc)] = enc
    struct.pack_into('<H', e, 64, nlen)
    e[66] = obj_type
    e[67] = color
    struct.pack_into('<I', e, 68, left)
    struct.pack_into('<I', e, 72, right)
    struct.pack_into('<I', e, 76, child)
    if clsid_hex:
        e[80:96] = bytes.fromhex(clsid_hex)
    struct.pack_into('<I', e, 116, start)
    struct.pack_into('<I', e, 120, size)
    return bytes(e)


def _ovba_record(id_, data):
    return struct.pack('<HI', id_, len(data)) + data


def _build_dir_stream(module_name: str) -> bytes:
    def u16(v): return struct.pack('<H', v)
    def u32(v): return struct.pack('<I', v)
    def s(v):   return v.encode('latin-1')

    r  = _ovba_record(0x0001, u32(0x00000001))   # SYSKIND Win32
    r += _ovba_record(0x0002, u32(0x0409))        # LCID
    r += _ovba_record(0x0014, u32(0x0409))        # LCIDINVOKE
    r += _ovba_record(0x0003, u16(1252))          # CODEPAGE
    r += _ovba_record(0x0004, s("VBAProject"))    # NAME
    r += _ovba_record(0x0005, b"")               # DOCSTRING
    r += _ovba_record(0x0040, b"")
    r += _ovba_record(0x0006, b"")               # HELPFILEPATH
    r += _ovba_record(0x003D, b"")
    r += _ovba_record(0x0007, u32(0))            # HELPCONTEXT
    r += _ovba_record(0x0008, u32(0))            # LIBFLAGS
    r += struct.pack('<HI', 0x0009, 4) + u32(0x61141F8C) + u16(0x000E)  # VERSION
    r += _ovba_record(0x000C, b"")               # CONSTANTS
    r += _ovba_record(0x003C, b"")
    # MODULES
    r += _ovba_record(0x000F, b"")
    r += _ovba_record(0x0013, u16(1))            # COUNT = 1
    r += _ovba_record(0x0028, u16(0x0077))       # COOKIE
    # MODULE
    r += _ovba_record(0x0019, s(module_name))
    r += _ovba_record(0x002F, module_name.encode('utf-16-le'))
    r += _ovba_record(0x001A, s(module_name))
    r += _ovba_record(0x0032, module_name.encode('utf-16-le'))
    r += _ovba_record(0x001C, b"")
    r += _ovba_record(0x0048, b"")
    r += _ovba_record(0x0031, u32(0))            # OFFSET into module stream
    r += _ovba_record(0x001E, u32(0x61141F8C))   # VERSION
    r += _ovba_record(0x0021, b"")               # TYPE procedural
    r += _ovba_record(0x0025, b"")               # READONLY
    r += struct.pack('<HI', 0x002B, 0)           # MODULE terminator
    r += struct.pack('<HI', 0x0010, 0)           # MODULES terminator
    return _compress_ovba(r)


def build_vba_project() -> io.BytesIO:
    """Create a minimal but valid vbaProject.bin OLE2 compound file."""
    src_bytes = VBA_SOURCE.encode('latin-1')
    module_compressed = _compress_ovba(src_bytes)
    dir_compressed    = _build_dir_stream(MODULE_NAME)

    _VBA_PROJECT_data = bytes([0xCC, 0x61]) + b'\x00' * 6

    PROJECT_data = (
        'ID="{00000000-0000-0000-0000-000000000000}"\r\n'
        f'Document=ThisWorkbook/&H00000000\r\n'
        f'Module={MODULE_NAME}\r\n'
        'HelpContextID="0"\r\n'
        'VersionCompatible32="393222000"\r\n'
        'CMG=""\r\nDPB=""\r\nGC=""\r\n'
    ).encode('latin-1')

    # All streams padded to ≥4096 bytes to bypass OLE2 mini-stream
    streams = [
        ('PROJECT',       _pad4096(PROJECT_data)),
        ('_VBA_PROJECT',  _pad4096(_VBA_PROJECT_data)),
        ('dir',           _pad4096(dir_compressed)),
        (MODULE_NAME,     _pad4096(module_compressed)),
        ('PROJECTwm',     _pad4096(b'')),
    ]

    # Sector layout:
    #   0 = FAT,  1 = dir sector 1 (SIDs 0-3),  2 = dir sector 2 (SIDs 4-7)
    #   3+ = stream data
    stream_sectors: dict[str, tuple[int, int, int]] = {}
    cur = 3
    for name, data in streams:
        nsec = len(data) // 512
        stream_sectors[name] = (cur, cur + nsec - 1, len(data))
        cur += nsec

    # FAT (sector 0): 128 entries × 4 bytes = 512 bytes
    fat = [FREESECT] * 128
    fat[0] = FATSECT        # sector 0 = FAT
    fat[1] = 2              # dir chain: 1 → 2
    fat[2] = ENDOFCHAIN     # end of dir chain
    for name, (start, end, _) in stream_sectors.items():
        for i in range(end - start):
            fat[start + i] = start + i + 1
        fat[end] = ENDOFCHAIN
    fat_sector = b''.join(struct.pack('<I', v) for v in fat)

    # Directory entries (128 bytes each):
    # SID 0 = Root Entry, SID 1 = VBA storage, SID 2 = PROJECT, SID 3 = PROJECTwm
    # SID 4 = _VBA_PROJECT, SID 5 = dir, SID 6 = MODULE, SID 7 = empty
    P = stream_sectors
    entries = [
        _dir_entry("Root Entry",  5, 1, NOSTREAM, NOSTREAM, 1,
                   "000204EF000000000000000000000046", NOSTREAM, 0),
        _dir_entry("VBA",         1, 1, NOSTREAM, 2, 4,
                   "EA7BAE70FB3B11CDA90300AA00510EA3", NOSTREAM, 0),
        _dir_entry("PROJECT",     2, 1, NOSTREAM, 3, NOSTREAM, None,
                   P['PROJECT'][0], P['PROJECT'][2]),
        _dir_entry("PROJECTwm",   2, 1, NOSTREAM, NOSTREAM, NOSTREAM, None,
                   P['PROJECTwm'][0], 0),
        _dir_entry("_VBA_PROJECT",2, 1, NOSTREAM, 5, NOSTREAM, None,
                   P['_VBA_PROJECT'][0], len(_VBA_PROJECT_data)),
        _dir_entry("dir",         2, 1, NOSTREAM, 6, NOSTREAM, None,
                   P['dir'][0], len(dir_compressed)),
        _dir_entry(MODULE_NAME,   2, 1, NOSTREAM, NOSTREAM, NOSTREAM, None,
                   P[MODULE_NAME][0], len(module_compressed)),
        _dir_entry("",            0, 1, NOSTREAM, NOSTREAM, NOSTREAM, None, NOSTREAM, 0),
    ]
    dir_sector1 = b''.join(entries[:4])
    dir_sector2 = b''.join(entries[4:8])

    # OLE2 header (512 bytes)
    hdr = bytearray(512)
    hdr[0:8]   = bytes.fromhex('D0CF11E0A1B11AE1')
    hdr[24:26] = struct.pack('<H', 0x003E)     # minor version
    hdr[26:28] = struct.pack('<H', 0x0003)     # major v3
    hdr[28:30] = struct.pack('<H', 0xFFFE)     # LE byte order
    hdr[30:32] = struct.pack('<H', 0x0009)     # sector size = 512 (2^9)
    hdr[32:34] = struct.pack('<H', 0x0006)     # mini sector = 64 (2^6)
    hdr[40:44] = struct.pack('<I', 0)          # dir sectors (v3=0)
    hdr[44:48] = struct.pack('<I', 1)          # FAT sectors count
    hdr[48:52] = struct.pack('<I', 1)          # first dir sector = 1
    hdr[52:56] = struct.pack('<I', 0)          # transaction sig
    hdr[56:60] = struct.pack('<I', 0x7FFFFFFF) # mini stream cutoff (huge → bypass)
    hdr[60:64] = struct.pack('<I', ENDOFCHAIN) # first mini FAT = none
    hdr[64:68] = struct.pack('<I', 0)          # mini FAT count
    hdr[68:72] = struct.pack('<I', ENDOFCHAIN) # first DIFAT = none
    hdr[72:76] = struct.pack('<I', 0)          # DIFAT sectors count
    # DIFAT entries at offsets 76..511 (109 entries × 4 bytes)
    struct.pack_into('<I', hdr, 76, 0)         # DIFAT[0] = sector 0 (FAT)
    for i in range(1, 109):
        struct.pack_into('<I', hdr, 76 + i * 4, FREESECT)

    out = bytearray()
    out += hdr
    out += fat_sector
    out += dir_sector1
    out += dir_sector2
    for _, data in streams:
        out += data

    return io.BytesIO(bytes(out))


# ── Data loading ───────────────────────────────────────────────────────────────

def load_leagues(leagues: list[str] | None, min_minutes: int) -> pd.DataFrame:
    if leagues is None:
        paths = sorted(p for p in WYSCOUT_DIR.glob("*.xlsx") if p.stem not in SKIP_FILES)
    else:
        paths = [WYSCOUT_DIR / f"{lg}.xlsx" for lg in leagues]

    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            print(f"  [warn] {path} not found — skipping")
            continue
        try:
            df = pd.read_excel(path)
        except Exception as e:
            print(f"  [warn] Could not read {path.name}: {e}")
            continue
        df = df.copy()
        df["_League"] = path.stem
        frames.append(df)
        print(f"  Loaded {path.stem}: {len(df)} rows")

    if not frames:
        raise RuntimeError(f"No Wyscout files found in {WYSCOUT_DIR}")

    raw = pd.concat(frames, ignore_index=True)

    mins_col = next((c for c in ["Minutes played", "MinutesPlayed", "Minutes"] if c in raw.columns), None)
    raw["_minutes"] = pd.to_numeric(raw[mins_col], errors="coerce").fillna(0) if mins_col else 0.0
    raw = raw[raw["_minutes"] >= min_minutes].copy()
    print(f"  → {len(raw)} players after {min_minutes}+ min filter")
    return raw.reset_index(drop=True)


def map_position(pos_str: str) -> str:
    if not isinstance(pos_str, str):
        return "Other"
    first = pos_str.split(",")[0].strip()
    return POS_MAP.get(first, "Other")


def add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    pos_col = next((c for c in ["Position", "Pos"] if c in df.columns), None)
    if pos_col:
        df["_pos_group"]    = df[pos_col].apply(map_position)
        df["_full_position"] = df[pos_col].fillna("Unknown")
    else:
        df["_pos_group"]    = "Other"
        df["_full_position"] = "Unknown"
    return df


# ── Index computation ──────────────────────────────────────────────────────────

def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    return series.rank(pct=True, ascending=ascending) * 100


def _get(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


def _weighted_score(components: list[tuple[pd.Series, float]]) -> pd.Series:
    total_w = sum(w for _, w in components)
    score   = sum(s * w for s, w in components)
    return score / total_w if total_w > 0 else score


def compute_sqs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SQS (Squad Quality Score) = Lamberts Index Score per position."""
    df = df.copy()
    df["_SQS_raw"] = np.nan
    df["_LI_score"] = np.nan

    for pos, blueprint in SQS_BLUEPRINTS.items():
        mask = df["_pos_group"] == pos
        if mask.sum() == 0:
            continue
        grp = df.loc[mask].copy()
        components = [(_pct_rank(_get(grp, col)), w) for col, w in blueprint]
        raw = _weighted_score(components)
        df.loc[mask, "_SQS_raw"] = raw.values

    for pos in SQS_BLUEPRINTS:
        mask = df["_pos_group"] == pos
        if mask.sum() == 0:
            continue
        df.loc[mask, "_LI_score"] = (
            df.loc[mask, "_SQS_raw"].rank(pct=True) * 100
        ).round(2)

    return df


def compute_defensive_indexes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute DIS, ADI, PADS, PDS for defensive positions only."""
    df = df.copy()
    for col in ["_DIS", "_ADI", "_PADS", "_PDS"]:
        df[col] = np.nan

    for pos in DEFENSIVE_POSITIONS:
        mask = df["_pos_group"] == pos
        if mask.sum() == 0:
            continue
        grp = df.loc[mask].copy()

        dis = _weighted_score([
            (_pct_rank(_get(grp, "Successful defensive actions per 90")), 3.0),
            (_pct_rank(_get(grp, "Defensive duels won, %")),             2.5),
            (_pct_rank(_get(grp, "Interceptions per 90")),               2.0),
            (_pct_rank(_get(grp, "PAdj Interceptions")),                 2.0),
            (_pct_rank(_get(grp, "Aerial duels won, %")),                1.5),
            (_pct_rank(_get(grp, "Shots blocked per 90")),               1.0),
        ]).round(2)

        adi = _pct_rank(
            _get(grp, "Aerial duels per 90") * _get(grp, "Aerial duels won, %") / 100.0
        ).round(2)

        pads = _weighted_score([
            (_pct_rank(_get(grp, "PAdj Interceptions")),      2.5),
            (_pct_rank(_get(grp, "PAdj Sliding tackles")),    2.5),
            (_pct_rank(_get(grp, "Shots blocked per 90")),    1.5),
            (_pct_rank(_get(grp, "Defensive duels per 90")),  1.0),
        ]).round(2)

        cost = (_get(grp, "Fouls per 90") * 0.6 +
                _get(grp, "Yellow cards per 90") * 0.3 +
                _get(grp, "Red cards per 90") * 0.1)
        pds = _pct_rank(cost, ascending=False).round(2)

        df.loc[mask, "_DIS"]  = dis.values
        df.loc[mask, "_ADI"]  = adi.values
        df.loc[mask, "_PADS"] = pads.values
        df.loc[mask, "_PDS"]  = pds.values

    return df


def compute_mv_rank(df: pd.DataFrame) -> pd.DataFrame:
    mv_col = next((c for c in ["Market value", "MarketValue"] if c in df.columns), None)
    mv = pd.to_numeric(df[mv_col], errors="coerce").fillna(0) if mv_col else pd.Series(0.0, index=df.index)
    df["_mkt_val"] = mv
    df["_mv_rank"]  = mv.rank(pct=True) * 100
    return df


def compute_lamberts_index(df: pd.DataFrame) -> pd.DataFrame:
    """Lamberts Index = LI Score Rank − Market Value Rank."""
    df["_LI"] = (df["_LI_score"] - df["_mv_rank"]).round(2)

    def tier(v: float) -> str:
        if v >= 30:  return "ELITE VALUE"
        if v >= 20:  return "HIGH VALUE"
        if v >= 10:  return "VALUE"
        if v >= 0:   return "FAIR VALUE"
        return "OVERPRICED"

    df["_tier"] = df["_LI"].apply(tier)
    return df


def compute_vs_hradec(df: pd.DataFrame) -> pd.DataFrame:
    vs_col, status_col = [], []
    for _, row in df.iterrows():
        pos    = row.get("_pos_group", "Other")
        hradec = HRADEC_SQUAD.get(pos, 50.0)
        li     = float(row.get("_LI_score", 50.0) or 50.0)
        gap    = round(li - hradec, 1)
        vs_col.append(gap)
        if gap > 0:       status_col.append("CLEAR UPGRADE")
        elif gap > -10:   status_col.append("ROTATIONAL / COVER")
        else:             status_col.append("DEPTH")
    df["_vs_hradec"] = vs_col
    df["_status"]    = status_col
    return df


def model_value(mkt: float, li_score: float) -> float:
    return round(mkt * max(li_score / 50.0, 0.1), 0)


def val_ratio_str(mkt: float, model: float) -> str:
    if mkt <= 0:
        return "N/A"
    return f"{model / mkt:.1f}×"


# ── Output table ──────────────────────────────────────────────────────────────

STAT_MAP = {
    # Defensive
    "Def Actions/90":   "Successful defensive actions per 90",
    "Def Duels/90":     "Defensive duels per 90",
    "Def Duel Won %":   "Defensive duels won, %",
    "Aerial/90":        "Aerial duels per 90",
    "Aerial Won %":     "Aerial duels won, %",
    "Intercept/90":     "Interceptions per 90",
    "PAdj Intercept":   "PAdj Interceptions",
    "Slides/90":        "Sliding tackles per 90",
    "PAdj Slides":      "PAdj Sliding tackles",
    "Blocks/90":        "Shots blocked per 90",
    "Fouls/90":         "Fouls per 90",
    "Yellow/90":        "Yellow cards per 90",
    # GK
    "Save %":           "Save rate, %",
    "Prevent Gls/90":   "Prevented goals per 90",
    "Exits/90":         "Exits per 90",
    # Passing
    "Prog Pass/90":     "Progressive passes per 90",
    "Acc Pass %":       "Accurate passes, %",
    "Key Pass/90":      "Key passes per 90",
    # Attacking
    "Goals/90":         "Goals per 90",
    "xG/90":            "xG per 90",
    "Assists/90":       "Assists per 90",
    "xA/90":            "xA per 90",
    "Dribbles/90":      "Dribbles per 90",
    "Prog Runs/90":     "Progressive runs per 90",
}

OUTPUT_COLS = [
    "Player", "Team", "League", "Pos", "Full Position", "Age",
    "Contract", "Exp?", "Mkt Val (€)", "Model Val (€)", "Val Ratio",
    "Tier", "LI Score", "Lamberts Index", "DIS", "ADI", "PADS", "PDS",
    "Status", "vs Hradec", "Minutes",
] + list(STAT_MAP.keys())


def build_master(df: pd.DataFrame) -> pd.DataFrame:
    contract_col = next((c for c in ["Contract expires", "ContractExpires"] if c in df.columns), None)
    rows: list[dict] = []

    for _, r in df.iterrows():
        mkt    = float(r.get("_mkt_val", 0) or 0)
        li_s   = float(r.get("_LI_score", 0) or 0)
        mod_v  = model_value(mkt, li_s)

        contract = r.get(contract_col) if contract_col else None
        if pd.notna(contract):
            try:
                contract = pd.to_datetime(contract).strftime("%Y-%m-%d")
            except Exception:
                contract = str(contract)
        else:
            contract = None

        exp_year = None
        if contract:
            try:
                exp_year = str(pd.to_datetime(contract).year)
            except Exception:
                pass

        def _f(col, default=0):
            v = r.get(col)
            try:
                return round(float(v), 2) if pd.notna(v) else default
            except Exception:
                return default

        row: dict = {
            "Player":         r.get("Player", ""),
            "Team":           r.get("Team", ""),
            "League":         r.get("_League", ""),
            "Pos":            r.get("_pos_group", ""),
            "Full Position":  r.get("_full_position", ""),
            "Age":            int(r.get("Age", 0)) if pd.notna(r.get("Age")) else "",
            "Contract":       contract,
            "Exp?":           exp_year,
            "Mkt Val (€)":    int(mkt) if mkt > 0 else 0,
            "Model Val (€)":  int(mod_v),
            "Val Ratio":      val_ratio_str(mkt, mod_v),
            "Tier":           r.get("_tier", ""),
            "LI Score":       round(li_s, 2),
            "Lamberts Index": round(float(r.get("_LI", 0) or 0), 2),
            "DIS":            _f("_DIS"),
            "ADI":            _f("_ADI"),
            "PADS":           _f("_PADS"),
            "PDS":            _f("_PDS"),
            "Status":         r.get("_status", ""),
            "vs Hradec":      r.get("_vs_hradec", 0),
            "Minutes":        int(r.get("_minutes", 0)),
        }
        for out_col, in_col in STAT_MAP.items():
            row[out_col] = _f(in_col)

        rows.append(row)

    master = pd.DataFrame(rows)
    master = master.sort_values("Lamberts Index", ascending=False).reset_index(drop=True)
    return master


# ── openpyxl helpers ──────────────────────────────────────────────────────────

def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _border() -> Border:
    thin = Side(style="thin", color="CCCCCC")
    return Border(left=thin, right=thin, top=thin, bottom=thin)


def _autofit(ws) -> None:
    for col_cells in ws.columns:
        try:
            max_len = max(
                len(str(col_cells[0].value or "")),
                *(len(str(c.value or "")) for c in col_cells[1:10]),
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)
        except Exception:
            pass


# ── xlsxwriter helpers ────────────────────────────────────────────────────────

def _xf(wb_x, bold=False, font_color="#000000", bg_color=None,
         size=9, italic=False, align="center", wrap=False, border=True):
    props: dict = {"font_size": size, "bold": bold, "italic": italic,
                   "font_color": font_color, "align": align, "valign": "vcenter"}
    if wrap:    props["text_wrap"] = True
    if bg_color: props["bg_color"] = bg_color; props["pattern"] = 1
    if border:  props.update({"border": 1, "border_color": "#CCCCCC"})
    return wb_x.add_format(props)


def write_data_sheet_fast(wb_x, sheet_name: str, title: str, subtitle: str,
                           df: pd.DataFrame) -> None:
    """Write a DataFrame as an xlsxwriter sheet with styled headers + AutoFilter."""
    if df.empty:
        ws = wb_x.add_worksheet(sheet_name)
        ws.write(0, 0, title)
        return

    ws = wb_x.add_worksheet(sheet_name)
    ws.set_zoom(85)

    cols = list(df.columns)
    n_cols = len(cols)

    # Row 1 — title
    fmt_title = _xf(wb_x, bold=True, font_color="#FFFFFF", bg_color=f"#{C['li_dark']}",
                    size=13, align="left", border=False)
    ws.merge_range(0, 0, 0, n_cols - 1, title, fmt_title)
    ws.set_row(0, 22)

    # Row 2 — subtitle
    fmt_sub = _xf(wb_x, italic=True, font_color=f"#{C['gold']}", bg_color=f"#{C['li_dark']}",
                  size=9, align="left", border=False)
    ws.merge_range(1, 0, 1, n_cols - 1, subtitle, fmt_sub)
    ws.set_row(1, 14)

    # Row 3 — column headers
    fmt_hdr = _xf(wb_x, bold=True, font_color="#FFFFFF", bg_color=f"#{C['header']}",
                  size=9, wrap=True)
    for j, col in enumerate(cols):
        ws.write(2, j, col, fmt_hdr)
    ws.set_row(2, 28)

    ws.freeze_panes(3, 0)
    ws.autofilter(2, 0, 2, n_cols - 1)

    TIER_BG   = {"ELITE VALUE": f"#{C['elite']}", "HIGH VALUE": f"#{C['high']}",
                 "VALUE": f"#{C['value']}", "FAIR VALUE": f"#{C['fair']}",
                 "OVERPRICED": f"#{C['over']}"}
    STATUS_BG = {"CLEAR UPGRADE": f"#{C['upgrade']}", "ROTATIONAL / COVER": f"#{C['rota']}",
                 "DEPTH": f"#{C['depth']}"}

    tier_idx   = cols.index("Tier")           if "Tier"           in cols else -1
    status_idx = cols.index("Status")         if "Status"         in cols else -1
    li_idx     = cols.index("Lamberts Index") if "Lamberts Index" in cols else -1
    lis_idx    = cols.index("LI Score")       if "LI Score"       in cols else -1

    fmt_even   = _xf(wb_x, bg_color=f"#{C['ice']}")
    fmt_odd    = _xf(wb_x, bg_color=f"#{C['white']}")
    fmt_bold_e = _xf(wb_x, bold=True, bg_color=f"#{C['ice']}")
    fmt_bold_o = _xf(wb_x, bold=True, bg_color=f"#{C['white']}")
    fmt_lp_e   = _xf(wb_x, bold=True, font_color=f"#{C['li_pos']}", bg_color=f"#{C['ice']}")
    fmt_ln_e   = _xf(wb_x, bold=True, font_color=f"#{C['li_neg']}", bg_color=f"#{C['ice']}")
    fmt_lp_o   = _xf(wb_x, bold=True, font_color=f"#{C['li_pos']}", bg_color=f"#{C['white']}")
    fmt_ln_o   = _xf(wb_x, bold=True, font_color=f"#{C['li_neg']}", bg_color=f"#{C['white']}")
    tier_fmts  = {k: _xf(wb_x, bold=True, font_color="#FFFFFF", bg_color=v)
                  for k, v in TIER_BG.items()}
    status_fmts = {k: _xf(wb_x, bold=True, font_color="#FFFFFF", bg_color=v)
                   for k, v in STATUS_BG.items()}

    for i, row_vals in enumerate(df.itertuples(index=False)):
        xrow  = i + 3
        even  = (i % 2 == 0)
        base  = fmt_even if even else fmt_odd
        bold  = fmt_bold_e if even else fmt_bold_o
        lp    = fmt_lp_e if even else fmt_lp_o
        ln    = fmt_ln_e if even else fmt_ln_o

        for j, val in enumerate(row_vals):
            if j == tier_idx:
                fmt = tier_fmts.get(str(val), base)
            elif j == status_idx:
                fmt = status_fmts.get(str(val), base)
            elif j == li_idx:
                try:  fmt = lp if float(val or 0) >= 0 else ln
                except Exception: fmt = base
            elif j == lis_idx:
                fmt = bold
            else:
                fmt = base

            if val is None or (isinstance(val, float) and np.isnan(val)):
                ws.write_blank(xrow, j, None, fmt)
            elif isinstance(val, (int, float)):
                ws.write_number(xrow, j, val, fmt)
            else:
                ws.write_string(xrow, j, str(val), fmt)

    for j, col in enumerate(cols):
        ws.set_column(j, j, min(max(len(col) + 2, 10), 22))


# ── README ────────────────────────────────────────────────────────────────────

def build_readme(ws, leagues: list[str], total: int, clear: int, budget: int) -> None:
    ws.title = "README"
    ws.sheet_view.showGridLines = False

    ws.append(["LAMBERTS INDEX — FULL PLAYER INTELLIGENCE REPORT  ·  FC Hradec Králové 2025–26"])
    ws.merge_cells("A1:T1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=15)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 30

    n_leagues = f"{len(leagues)} leagues" if len(leagues) > 5 else " + ".join(leagues)
    ws.append([f"Waltzing Analytics  ·  Lamberts Index methodology  ·  {n_leagues}  ·  "
               f"Positions: GK · CB · FB · DM · CM · W · FW  ·  Budget ≤ €{budget:,}"])
    ws.merge_cells("A2:T2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=10)
    ws["A2"].fill = _fill(C["li_dark"])
    ws.row_dimensions[2].height = 18
    ws.append([None])

    ws.append([None, "WORKBOOK STRUCTURE"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])
    ws.row_dimensions[ws.max_row].height = 20

    ws.append([None, "Sheet", "Contents"])
    hdr = ws.max_row
    for col in "BC":
        c = ws[f"{col}{hdr}"]
        c.font = Font(bold=True, color=C["white"])
        c.fill = _fill(C["header"])
        c.alignment = Alignment(horizontal="left")

    structure = [
        ("README",             "This guide — methodology and index definitions"),
        ("Filter Panel",       "Macro controls — pick Position / Tier / Status then click Apply Filters"),
        ("Priority List",      f"{clear} clear upgrades across all positions, ranked by Lamberts Index"),
        ("Elite Picks",        "Players with Lamberts Index ≥ 30 — strongest buy signals"),
        ("Lamberts Analysis",  "Tier and position breakdown summary statistics"),
        ("Top 5",              "Best 5 per position by Lamberts Index"),
        ("All Players",        f"Full {total} candidate database — use filters or Filter Panel macro"),
        ("GK",                 "Goalkeeper targets"),
        ("CB",                 "Centre-back targets"),
        ("FB",                 "Full-back targets"),
        ("DM",                 "Defensive mid targets"),
        ("CM",                 "Central mid targets"),
        ("W",                  "Winger targets"),
        ("FW",                 "Forward targets"),
        ("Expiring 2026",      "Contract opportunities — free or discounted"),
        ("Budget Planner",     f"Build squad within €{budget:,}"),
        ("Squad",              "Hradec Králové current quality benchmarks"),
    ]
    for sheet, desc in structure:
        ws.append([None, sheet, desc])
        row = ws.max_row
        ws[f"B{row}"].font = Font(bold=True, color=C["navy"])

    ws.append([None])
    ws.append([None, "INDEX DEFINITIONS"])
    ws[f"B{ws.max_row}"].font = Font(bold=True, size=11, color=C["navy"])

    defs = [
        ("LI Score (0–100)",
         "Lamberts Index Score: position-specific SQS percentile rank based on weighted blueprint metrics. "
         "Analogous to SQS in the original Lamberts model — 100 = best in position across all leagues."),
        ("Lamberts Index",
         "LI Score Rank − Market Value Rank. Positive = undervalued vs. the market. "
         "ELITE ≥30 · HIGH ≥20 · VALUE ≥10 · FAIR 0–9 · OVERPRICED <0"),
        ("DIS — Defensive Impact Score (def positions only)",
         "Weighted rank: Def actions/90 ×3 · Def duels won% ×2.5 · Interceptions/90 ×2 · "
         "PAdj Interceptions ×2 · Aerial won% ×1.5 · Blocks/90 ×1"),
        ("ADI — Aerial Dominance Index (def positions only)",
         "Aerial duels/90 × aerial won% / 100, ranked 0–100. Rewards volume + quality of aerial work."),
        ("PADS — Pressure-Adjusted Defensive Score (def positions only)",
         "Uses PAdj metrics to strip possession-context bias: PAdj Interceptions ×2.5 · "
         "PAdj Slides ×2.5 · Blocks/90 ×1.5 · Def duels/90 ×1"),
        ("PDS — Positional Discipline Score (def positions only)",
         "Inverted rank of foul cost (Fouls×0.6 + Yellow×0.3 + Red×0.1). High PDS = clean defending."),
        ("vs Hradec",
         "LI Score − Hradec starter benchmark at same position. "
         ">0 = CLEAR UPGRADE · −10..0 = ROTATIONAL · <−10 = DEPTH"),
    ]
    for term, desc in defs:
        ws.append([None, term, desc])
        row = ws.max_row
        ws[f"B{row}"].font = Font(bold=True)
        ws[f"B{row}"].fill = _fill(C["ice"])

    ws.column_dimensions["A"].width = 3
    ws.column_dimensions["B"].width = 38
    ws.column_dimensions["C"].width = 95


# ── Filter Panel sheet ────────────────────────────────────────────────────────

def build_filter_panel(ws, positions: list[str], tiers: list[str], statuses: list[str]) -> None:
    ws.title = "Filter Panel"
    ws.sheet_view.showGridLines = False

    ws.append(["LAMBERTS INDEX — FILTER PANEL"])
    ws.merge_cells("A1:F1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=15)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 30

    ws.append(["Set filters below, then click Apply Filters (or run the macro from the Developer tab)."])
    ws.merge_cells("A2:F2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=10)
    ws["A2"].fill = _fill(C["li_dark"])
    ws.row_dimensions[2].height = 18
    ws.append([None])

    def labeled_row(label, value, note=""):
        ws.append([label, value, note])
        row = ws.max_row
        ws[f"A{row}"].font = Font(bold=True, size=11)
        ws[f"A{row}"].fill = _fill(C["ice"])
        ws[f"A{row}"].alignment = Alignment(horizontal="right", vertical="center")
        ws[f"B{row}"].font = Font(size=11)
        ws[f"B{row}"].alignment = Alignment(horizontal="left", vertical="center")
        ws[f"C{row}"].font = Font(italic=True, size=9, color=C["fair"])
        ws.row_dimensions[row].height = 20

    labeled_row("Position →", "All",   "Options: All · GK · CB · FB · DM · CM · W · FW")
    labeled_row("Tier →",     "All",   "Options: All · ELITE VALUE · HIGH VALUE · VALUE · FAIR VALUE · OVERPRICED")
    labeled_row("Status →",   "All",   "Options: All · CLEAR UPGRADE · ROTATIONAL / COVER · DEPTH")
    ws.append([None])

    # Instructions
    ws.append(["HOW TO USE"])
    ws[f"A{ws.max_row}"].font = Font(bold=True, size=11)
    instructions = [
        "1. Change the values in column B (cells B4, B5, B6) to your desired filter.",
        '2. Use "All" to leave a filter unset.',
        "3. Open the Developer tab → Macros → Run 'LambertsFilters.ApplyFilters'",
        "   OR press Alt+F8, select ApplyFilters, click Run.",
        "4. All data sheets will be filtered simultaneously.",
        "5. To clear all filters: run 'LambertsFilters.ClearAllFilters'",
    ]
    for line in instructions:
        ws.append([line])
        ws[f"A{ws.max_row}"].font = Font(size=10)
    ws.append([None])

    ws.append(["VALID FILTER VALUES"])
    ws[f"A{ws.max_row}"].font = Font(bold=True, size=11)

    ws.append(["Position:", ", ".join(["All"] + sorted(positions))])
    ws.append(["Tier:",     ", ".join(["All"] + tiers)])
    ws.append(["Status:",   ", ".join(["All"] + statuses)])
    for r_idx in range(ws.max_row - 2, ws.max_row + 1):
        ws[f"A{r_idx}"].font = Font(bold=True)
        ws[f"B{r_idx}"].font = Font(size=9)

    ws.column_dimensions["A"].width = 20
    ws.column_dimensions["B"].width = 16
    ws.column_dimensions["C"].width = 70


# ── Lamberts Analysis sheet ───────────────────────────────────────────────────

def build_analysis(ws, master: pd.DataFrame) -> None:
    ws.title = "Lamberts Analysis"
    ws.sheet_view.showGridLines = False

    ws.append(["LAMBERTS INDEX — FULL ANALYSIS BREAKDOWN"])
    ws.merge_cells("A1:N1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 22

    ws.append(["Lamberts Index = LI Score Rank − Market Value Rank  ·  ELITE ≥30  ·  HIGH ≥20  ·  VALUE ≥10  ·  FAIR 0–9  ·  OVER <0"])
    ws.merge_cells("A2:N2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["li_dark"])
    ws.row_dimensions[2].height = 14
    ws.append([None])

    ws.append(["OVERALL STATISTICS"])
    ws[f"A{ws.max_row}"].font = Font(bold=True, size=11)

    li = master["Lamberts Index"]
    overall = [
        ("Total Players",  len(master)),
        ("Mean LI",        round(li.mean(), 2)),
        ("Median LI",      round(li.median(), 2)),
        ("Max LI",         round(li.max(), 2)),
        ("Min LI",         round(li.min(), 2)),
        ("Std Dev",        round(li.std(), 2)),
    ]

    pos_stats = master.groupby("Pos")["LI Score"].mean().round(2)

    ws.append(["Metric", "Value", None, "Tier", "Count", "%", None, "Position", "Avg LI Score"])
    hdr = ws.max_row
    for cell in ws[hdr]:
        if cell.value:
            cell.font = Font(bold=True, color=C["white"])
            cell.fill = _fill(C["header"])

    tier_order = ["ELITE VALUE", "HIGH VALUE", "VALUE", "FAIR VALUE", "OVERPRICED"]
    tier_counts = master["Tier"].value_counts()
    pos_order   = ["GK", "CB", "FB", "DM", "CM", "W", "FW"]

    for i, (metric, val) in enumerate(overall):
        ws.append([metric, val])
        row = ws.max_row
        tier_name = tier_order[i] if i < len(tier_order) else ""
        cnt = int(tier_counts.get(tier_name, 0))
        pct = round(cnt / len(master) * 100, 1) if len(master) > 0 else 0
        ws.cell(row, 4).value = tier_name
        ws.cell(row, 5).value = cnt
        ws.cell(row, 6).value = f"{pct}%"
        if i < len(pos_order):
            p = pos_order[i]
            ws.cell(row, 8).value = p
            ws.cell(row, 9).value = float(pos_stats.get(p, 0))

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 12
    ws.column_dimensions["D"].width = 18
    for col in "EFH": ws.column_dimensions[col].width = 12
    ws.column_dimensions["I"].width = 16


# ── Top 5 per position ────────────────────────────────────────────────────────

def build_top5(ws, master: pd.DataFrame) -> None:
    ws.title = "Top 5"
    ws.sheet_view.showGridLines = False

    ws.append(["LAMBERTS INDEX — TOP 5 PER POSITION  ·  Ranked by Lamberts Index"])
    ws.merge_cells("A1:N1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 22

    ws.append(["Highest Lamberts Index per position across all scouted leagues"])
    ws.merge_cells("A2:N2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["li_dark"])

    medals = ["1.", "2.", "3.", "4.", "5."]

    for pos, label in POS_LABELS.items():
        grp = master[master["Pos"] == pos].head(5)
        if grp.empty:
            continue

        ws.append([f"  {label}"])
        ws.merge_cells(start_row=ws.max_row, start_column=1, end_row=ws.max_row, end_column=14)
        ws[f"A{ws.max_row}"].font = Font(bold=True, color=C["white"], size=10)
        ws[f"A{ws.max_row}"].fill = _fill(C["navy"])
        ws.row_dimensions[ws.max_row].height = 18

        ws.append(["#", "Player", "Team", "League", "Age", "Contract",
                   "Mkt Val (€)", "LI Score", "Lamberts Index", "DIS", "ADI", "PADS", "PDS", "Status"])
        hdr = ws.max_row
        for cell in ws[hdr]:
            cell.font = Font(bold=True, color=C["white"], size=9)
            cell.fill = _fill(C["header"])
            cell.alignment = Alignment(horizontal="center")

        for i, (_, r) in enumerate(grp.iterrows()):
            ws.append([
                medals[i], r["Player"], r["Team"], r["League"],
                r["Age"], r["Contract"], r["Mkt Val (€)"],
                r["LI Score"], r["Lamberts Index"],
                r.get("DIS", ""), r.get("ADI", ""), r.get("PADS", ""), r.get("PDS", ""),
                r["Status"],
            ])
            dr = ws.max_row
            bg = C["ice"] if i % 2 == 0 else C["white"]
            for cell in ws[dr]:
                cell.font = Font(size=9)
                cell.fill = _fill(bg)
                cell.alignment = Alignment(horizontal="center")
            ws.cell(dr, 1).font = Font(bold=True, size=10)

        ws.append([None])

    _autofit(ws)


# ── Budget Planner ────────────────────────────────────────────────────────────

def build_budget_planner(ws, master: pd.DataFrame, budget: int) -> None:
    ws.title = "Budget Planner"
    ws.sheet_view.showGridLines = False

    ws.append([f"LAMBERTS INDEX — BUDGET PLANNER  ·  FC Hradec Králové  ·  Cap: €{budget:,}"])
    ws.merge_cells("A1:O1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 22

    ws.append(["ELITE VALUE + HIGH VALUE clear upgrades sorted by market value (cheapest first)."])
    ws.merge_cells("A2:O2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["li_dark"])

    ws.append([None, None, None, None, "Total Budget", budget])

    eligible = master[
        master["Tier"].isin(["ELITE VALUE", "HIGH VALUE"]) &
        (master["Status"] == "CLEAR UPGRADE")
    ].sort_values("Mkt Val (€)").reset_index(drop=True)

    cols = ["#", "Player", "Pos", "Team", "League", "Age", "Contract",
            "Mkt Val (€)", "Model Val (€)", "Val Ratio", "LI Score", "Lamberts Index",
            "vs Hradec", "Running Total (€)"]

    ws.append(cols)
    hdr = ws.max_row
    for cell in ws[hdr]:
        cell.font = Font(bold=True, color=C["white"], size=9)
        cell.fill = _fill(C["header"])
        cell.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.row_dimensions[hdr].height = 28

    running = 0
    for i, (_, r) in enumerate(eligible.iterrows(), 1):
        mv = int(r["Mkt Val (€)"])
        running += mv
        over = running > budget
        ws.append([
            i, r["Player"], r["Pos"], r["Team"], r["League"],
            r["Age"], r["Contract"],
            mv, int(r["Model Val (€)"]), r["Val Ratio"],
            r["LI Score"], r["Lamberts Index"],
            r["vs Hradec"], running,
        ])
        dr = ws.max_row
        for cell in ws[dr]:
            cell.font = Font(size=9, color="922B21" if over else "000000")
            cell.fill = _fill("FDE8E8" if over else (C["ice"] if i % 2 == 0 else C["white"]))
            cell.alignment = Alignment(horizontal="center")

    ws.freeze_panes = f"A{hdr + 1}"
    _autofit(ws)


# ── Squad benchmarks ──────────────────────────────────────────────────────────

def build_squad(ws) -> None:
    ws.title = "Squad"
    ws.sheet_view.showGridLines = False

    ws.append(["FC HRADEC KRÁLOVÉ — SQUAD QUALITY BENCHMARKS"])
    ws.merge_cells("A1:L1")
    ws["A1"].font = Font(bold=True, color=C["white"], size=13)
    ws["A1"].fill = _fill(C["li_dark"])
    ws.row_dimensions[1].height = 22

    ws.append(["Czech top-flight LI Score percentile  ·  Basis for vs Hradec calculation"])
    ws.merge_cells("A2:L2")
    ws["A2"].font = Font(italic=True, color=C["gold"], size=9)
    ws["A2"].fill = _fill(C["li_dark"])

    ws.append([None])
    ws.append(["SQUAD QUALITY RATINGS"])
    ws[f"A{ws.max_row}"].font = Font(bold=True, size=11)

    ws.append(["Position", "LI Score Benchmark", "Note"])
    hdr = ws.max_row
    for cell in ws[hdr]:
        if cell.value:
            cell.font = Font(bold=True, color=C["white"])
            cell.fill = _fill(C["header"])
            cell.alignment = Alignment(horizontal="center")

    pos_notes = {
        "GK": "Adam Zadrazil — starter",
        "CB": "Daniel Horak — weakest starter",
        "FB": "Martin Suchomel — starter",
        "DM": "Jakub Elbel — starter",
        "CM": "Median starter quality",
        "W":  "Weakest winger in squad",
        "FW": "Attacking dept. reference",
    }
    for pos, quality in HRADEC_SQUAD.items():
        ws.append([pos, quality, pos_notes.get(pos, "")])
        row = ws.max_row
        ws.cell(row, 1).font = Font(bold=True)
        ws.cell(row, 2).alignment = Alignment(horizontal="center")

    ws.column_dimensions["A"].width = 12
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 40


# ── Main ───────────────────────────────────────────────────────────────────────

def run(leagues: list[str] | None, min_minutes: int, max_age: int,
        budget: int, output: Path) -> None:
    print(f"\n{'='*66}")
    print("  LAMBERTS INDEX — Full Player Intelligence Report")
    print(f"  Positions: GK · CB · FB · DM · CM · W · FW")
    print(f"  Leagues: {'ALL' if leagues is None else leagues}")
    print(f"  Min minutes: {min_minutes}  |  Max age: {max_age}")
    print(f"{'='*66}\n")

    print("Loading Wyscout files…")
    raw = load_leagues(leagues, min_minutes)

    if "Age" in raw.columns:
        raw = raw[pd.to_numeric(raw["Age"], errors="coerce").fillna(99) <= max_age]
        print(f"  → {len(raw)} players after age ≤ {max_age}")

    raw = add_position_group(raw)
    raw = raw[raw["_pos_group"].isin(ALL_POSITIONS)].copy()
    print(f"  → {len(raw)} players across all positions")

    print("Computing Lamberts Index scores…")
    raw = compute_sqs(raw)
    raw = compute_defensive_indexes(raw)
    raw = compute_mv_rank(raw)
    raw = compute_lamberts_index(raw)
    raw = compute_vs_hradec(raw)

    print("Building master table…")
    master = build_master(raw)
    master_out = master[[c for c in OUTPUT_COLS if c in master.columns]]

    total  = len(master)
    clear  = int((master["Status"] == "CLEAR UPGRADE").sum())
    elite  = int((master["Tier"] == "ELITE VALUE").sum())
    print(f"  → {total} players total | {clear} clear upgrades | {elite} ELITE VALUE")

    league_list = leagues if leagues else sorted(master["League"].unique().tolist())
    positions   = sorted(master["Pos"].unique().tolist())
    tiers       = ["ELITE VALUE", "HIGH VALUE", "VALUE", "FAIR VALUE", "OVERPRICED"]
    statuses    = ["CLEAR UPGRADE", "ROTATIONAL / COVER", "DEPTH"]

    print(f"\nBuilding VBA macro project…")
    vba_stream = build_vba_project()

    print(f"Writing workbook → {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # Force .xlsm extension
    if output.suffix.lower() != ".xlsm":
        output = output.with_suffix(".xlsm")

    xls_path = output.with_suffix(".tmp.xlsm")

    # ── Phase 1: xlsxwriter data sheets ───────────────────────────────────────
    print("  Building data sheets (xlsxwriter)…")
    wb_x = xlsxwriter.Workbook(str(xls_path), {"constant_memory": False})
    wb_x.set_vba_name("VBAProject")

    prio_df = master_out[master_out["Status"] == "CLEAR UPGRADE"].copy()
    print(f"    Priority List ({len(prio_df)} rows)…")
    write_data_sheet_fast(wb_x, "Priority List",
        f"LAMBERTS INDEX — PRIORITY LIST  ·  {len(prio_df)} Clear Upgrades",
        "All outperform current Hradec starters at same position  ·  Sorted by Lamberts Index",
        prio_df)

    elite_df = master_out[master_out["Tier"] == "ELITE VALUE"].copy()
    print(f"    Elite Picks ({len(elite_df)} rows)…")
    write_data_sheet_fast(wb_x, "Elite Picks",
        f"LAMBERTS INDEX — ELITE VALUE  ·  {len(elite_df)} Players with LI ≥ 30",
        "Strongest buy signals across all positions  ·  Sorted by Lamberts Index",
        elite_df)

    print(f"    All Players ({total} rows)…")
    write_data_sheet_fast(wb_x, "All Players",
        f"LAMBERTS INDEX — ALL PLAYERS  ·  {total} Candidates  ·  {len(league_list)} Leagues",
        "Use column filter dropdowns or Filter Panel macro to narrow results",
        master_out)

    for pos in ["GK", "CB", "FB", "DM", "CM", "W", "FW"]:
        grp = master_out[master_out["Pos"] == pos].copy()
        print(f"    {pos} ({len(grp)} rows)…")
        write_data_sheet_fast(wb_x, pos,
            f"LAMBERTS INDEX — {POS_LABELS[pos]}  ·  {len(grp)} Candidates",
            "Sorted by Lamberts Index",
            grp)

    exp_df = master_out[master_out["Exp?"].astype(str) == "2026"].copy()
    print(f"    Expiring 2026 ({len(exp_df)} rows)…")
    write_data_sheet_fast(wb_x, "Expiring 2026",
        f"EXPIRING CONTRACTS 2026  ·  {len(exp_df)} Players",
        "Free agent or discounted transfer opportunities",
        exp_df)

    wb_x.add_vba_project(vba_stream, is_stream=True)
    wb_x.close()
    print("  xlsxwriter sheets done.")

    # ── Phase 2: openpyxl summary sheets ──────────────────────────────────────
    print("  Building summary sheets (openpyxl)…")
    from openpyxl import load_workbook
    wb = load_workbook(str(xls_path), keep_vba=True)

    ws_readme = wb.create_sheet("README", 0)
    build_readme(ws_readme, league_list, total, clear, budget)

    ws_fp = wb.create_sheet("Filter Panel", 1)
    build_filter_panel(ws_fp, positions, tiers, statuses)

    ws_analysis = wb.create_sheet("Lamberts Analysis", 4)
    build_analysis(ws_analysis, master)

    ws_top5 = wb.create_sheet("Top 5", 5)
    build_top5(ws_top5, master_out)

    ws_budget = wb.create_sheet("Budget Planner")
    build_budget_planner(ws_budget, master, budget)

    ws_squad = wb.create_sheet("Squad")
    build_squad(ws_squad)

    if "Sheet1" in wb.sheetnames:
        del wb["Sheet1"]

    wb.save(str(output))
    xls_path.unlink(missing_ok=True)

    size_mb = output.stat().st_size / 1_048_576
    print(f"\nDone. {size_mb:.1f} MB → {output.resolve()}")
    print(f"  Sheets: {', '.join(wb.sheetnames)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lamberts Index — Full Player Intelligence Report from Wyscout data"
    )
    parser.add_argument("--leagues",     nargs="+", default=None,
                        help="League names (without .xlsx). Omit for ALL leagues.")
    parser.add_argument("--min-minutes", type=int,  default=DEFAULT_MIN_MINUTES)
    parser.add_argument("--max-age",     type=int,  default=DEFAULT_MAX_AGE)
    parser.add_argument("--budget",      type=int,  default=DEFAULT_BUDGET)
    parser.add_argument("--output",      type=Path,
                        default=ROOT / "data" / "Lamberts_Index_Full_Report.xlsm")
    args = parser.parse_args()
    run(
        leagues=args.leagues,
        min_minutes=args.min_minutes,
        max_age=args.max_age,
        budget=args.budget,
        output=args.output,
    )
