"""Phase 1c: regex event tags for headlines. Returns one 0/1 column per event."""
import re
import warnings

import pandas as pd

# Simple keyword patterns (case-insensitive). Order/names are the dummy column names.
EVENT_PATTERNS = {
    "ev_earnings_beat": r"\b(beats?|tops?|top(?:ped)?|surpass(?:es|ed)?|above|exceed(?:s|ed)?)\b.{0,40}\b(estimates?|expectations?|consensus|forecasts?)\b|\bearnings beat\b|\bbeat(?:s)? (?:on|q[1-4])\b",
    "ev_earnings_miss": r"\b(miss(?:es|ed)?|falls? short|below|lags?|trails?)\b.{0,40}\b(estimates?|expectations?|consensus|forecasts?)\b|\bearnings miss\b",
    "ev_guidance": r"\bguidance\b|\b(raises?|lowers?|cuts?|boosts?|issues?|reaffirms?|updates?|sees|outlook)\b.{0,30}\b(outlook|forecast|guidance|view)\b",
    "ev_buyback": r"\bbuy-?backs?\b|\bshare repurchase\b|\brepurchase (?:program|plan|authorization)\b|\brepurchases? (?:up to )?\$",
    "ev_dividend": r"\bdividends?\b|\bpayout\b|\bex-dividend\b",
    "ev_merger_acq": r"\bmerger\b|\bacqui(?:re|res|red|sition|sitions)\b|\btakeover\b|\bbuyout\b|\bto buy\b.{0,30}\b(for|in)\b.{0,20}\$|\bdeal to (?:buy|acquire)\b|\bmerges?\b",
    "ev_fda_approval": r"\bfda\b.{0,40}\b(approv(?:al|als|es|ed)|clears?|cleared|nod|green light)\b|\bapproved by (?:the )?fda\b",
    "ev_fda_rejection": r"\bfda\b.{0,40}\b(reject(?:s|ed|ion)?|refuses?|complete response letter|crl|declines?|denies|denied|delays?)\b",
    "ev_investigation_lawsuit": r"\b(investigation|investigates?|probe|subpoena|lawsuit|sues?|sued|class action|litigation|settle(?:s|d|ment)?|sec charges|indict(?:ed|ment)?)\b",
    "ev_downgrade": r"\bdowngrad(?:e|es|ed|ing)\b|\blowers? (?:rating|to (?:sell|underperform|neutral|hold))\b|\bcut to (?:sell|underperform|neutral|hold)\b",
    "ev_upgrade": r"\bupgrad(?:e|es|ed|ing)\b|\braises? (?:rating|to (?:buy|outperform|overweight))\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in EVENT_PATTERNS.items()}
EVENT_COLS = list(EVENT_PATTERNS)


def tag_events(headlines: pd.Series) -> pd.DataFrame:
    """0/1 int8 dummy per event, same index as `headlines`."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)      # "pattern has match groups" is harmless here
        return pd.DataFrame({k: headlines.str.contains(rx, regex=True).astype("int8")
                             for k, rx in _COMPILED.items()}, index=headlines.index)


if __name__ == "__main__":
    demo = pd.Series(["Apple Beats Q3 EPS Estimates", "Pfizer Misses Revenue Estimates",
                      "FDA Approves Merck's Keytruda", "FDA Rejects Drug Application",
                      "Analyst Downgrades Ford to Sell", "Company Announces $5B Buyback",
                      "Firm raises full-year guidance", "Cisco declares quarterly dividend",
                      "Microsoft to acquire Activision", "SEC opens probe into Tesla",
                      "Stocks that hit 52-week highs"])
    print(pd.concat([demo.rename("h"), tag_events(demo)], axis=1).set_index("h").to_string())
