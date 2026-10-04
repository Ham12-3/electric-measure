"""Calendar features. These are always known in advance, so they're safe at any horizon.

Computed in UK local time: people's routines follow the clock on the wall, so the
07:00 morning ramp happens at 07:00 local, which is 06:00 UTC in summer.
"""

from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

UK_TZ = "Europe/London"


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    local = index.tz_convert(UK_TZ)
    dates = local.normalize().tz_localize(None).date
    years = range(local.year.min() - 1, local.year.max() + 2)
    # England & Wales holidays affect ~90% of demand; Scotland's differ on a few days
    # (2 Jan, St Andrew's Day, the August holiday) so they get their own flag.
    ew = holidays.UK(subdiv="ENG", years=years)
    sct = holidays.UK(subdiv="SCT", years=years)

    doy = local.dayofyear.to_numpy()
    month, day = local.month.to_numpy(), local.day.to_numpy()
    out = pd.DataFrame({
        "hour": local.hour,
        "dow": local.dayofweek,
        "month": month,
        "is_weekend": (local.dayofweek >= 5).astype(int),
        "is_holiday_ew": np.array([d in ew for d in dates], dtype=int),
        "is_holiday_sct": np.array([d in sct for d in dates], dtype=int),
        # Between Christmas Eve and New Year's Day many businesses close, so demand looks
        # like a long holiday even on normal working days.
        "is_christmas_period": (((month == 12) & (day >= 24)) | ((month == 1) & (day == 1))).astype(int),
        # Smooth position in the year (so 31 Dec and 1 Jan are neighbours, not opposite ends).
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
    }, index=index)
    return out
