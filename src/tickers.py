"""Ticker aliasing: map a raw symbol on a given date to one canonical ticker per company.

Each rule is (raw_symbol, before_date, target):
  rows with this raw symbol dated BEFORE before_date are relabelled to target;
  target None = the symbol belonged to a different company then -> drop the row.
Rows on/after before_date keep the raw symbol. ALL = every pre-2023 date.
Used for both FNSPID news symbols and historical S&P 500 membership snapshots.
Checked against FNSPID per-year symbol counts (data/processed/fnspid_symbol_year_counts.csv).
"""
import pandas as pd

ALL = "2100-01-01"

RULES = [
    # renames into CURRENT S&P 500 tickers
    ("FB", ALL, "META"), ("ANTM", ALL, "ELV"), ("BHGE", ALL, "BKR"), ("HRS", ALL, "LHX"),
    ("JEC", ALL, "J"), ("TMK", ALL, "GL"), ("BBT", ALL, "TFC"), ("WLTW", ALL, "WTW"),
    ("SYMC", ALL, "GEN"), ("NLOK", ALL, "GEN"), ("UTX", ALL, "RTX"), ("MYL", ALL, "VTRS"),
    ("ARNC", "2020-04-06", "HWM"), ("DWDP", ALL, "DD"), ("BLL", ALL, "BALL"),
    ("HCP", ALL, "DOC"), ("PEAK", ALL, "DOC"), ("FLT", ALL, "CPAY"), ("ABC", ALL, "COR"),
    ("PKI", ALL, "RVTY"), ("RE", ALL, "EG"), ("MMC", ALL, "MRSH"), ("ACE", ALL, "CB"),
    ("CBG", ALL, "CBRE"), ("HCN", ALL, "WELL"), ("DLPH", ALL, "APTV"), ("COH", ALL, "TPR"),
    ("PCLN", ALL, "BKNG"), ("MHFI", ALL, "SPGI"), ("NU", ALL, "ES"), ("ZMH", ALL, "ZBH"),
    ("WR", ALL, "EVRG"), ("Q", ALL, "IQV"),
    ("IR", "2020-03-02", "TT"),                         # old Ingersoll-Rand plc became Trane (TT)
    ("NWSA", "2013-06-28", "FOXA"), ("NWS", "2013-06-28", "FOX"),   # pre-split News Corp = Fox
    # recycled symbols: before the date the symbol was a DIFFERENT company -> drop
    ("WTW", "2022-01-10", None),     # Weight Watchers
    ("DOC", "2024-03-01", None),     # Physicians Realty Trust
    ("COR", "2023-08-30", None),     # CoreSite Realty
    ("GEN", "2022-11-08", None),     # Genesis Healthcare
    ("ES", "2015-04-30", None),      # pre-Eversource
    ("CZR", "2020-07-21", None),     # old Caesars Entertainment (Eldorado took the symbol)
    # renames among companies later REMOVED from the index (used for historical membership)
    ("KORS", ALL, "CPRI"), ("COG", ALL, "CTRA"), ("CTL", ALL, "LUMN"), ("LB", ALL, "BBWI"),
    ("DISCA", ALL, "WBD"), ("DISCK", ALL, "WBD"), ("CDAY", ALL, "DAY"), ("CBS", ALL, "PARA"),
    ("VIAC", ALL, "PARA"), ("YHOO", ALL, "AABA"), ("ERI", ALL, "CZR"),
]

# dual share classes of one company: company news is attached to both tickers
SHARE_CLASSES = [("GOOG", "GOOGL"), ("FOX", "FOXA"), ("NWS", "NWSA")]


def canonical(symbols: pd.Series, dates: pd.Series) -> pd.Series:
    """Vectorised: canonical ticker for each (raw symbol, date); NaN = drop the row."""
    out = symbols.astype(object).copy()
    d = pd.to_datetime(dates)
    for raw, before, target in RULES:
        m = (symbols == raw) & (d < pd.Timestamp(before))
        out.loc[m] = target if target is not None else None
    return out


def expand_share_classes(df: pd.DataFrame, col="ticker") -> pd.DataFrame:
    """Copy each row of one share class to its sibling class (caller dedupes)."""
    extra = []
    for a, b in SHARE_CLASSES:
        for src, dst in [(a, b), (b, a)]:
            e = df.loc[df[col] == src].copy()
            e.loc[:, col] = dst
            extra.append(e)
    return pd.concat([df] + extra, ignore_index=True)
