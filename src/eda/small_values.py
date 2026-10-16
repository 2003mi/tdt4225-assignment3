"""Extra step 1: budgets and revenues under $1,000.

The numeric summary showed 245 budgets and 155 revenues between 1 and 999
USD. This step lists them, sorted so the most telling cases come first:
  - small budgets, sorted by revenue (highest first): a $113 budget next to
    a $115M revenue is obviously not in dollars,
  - small revenues, sorted by budget (highest first).

Finding: the values mix units. Some are in thousands (Little Witch Academia:
625 = $625k, Kickstarter-funded), some in millions (The Karate Kid, Part II:
113 = $113M), and many are placeholders (1). No single rule can rescale them,
so values under 1000 are set to null - each column on its own, since a movie
can have a broken budget but a real revenue.
"""
import pandas as pd

from .common import section, short, show

LIMIT = 1000
ROWS = 20


def run(meta: pd.DataFrame) -> None:
    """meta: metadata from core.eda_metadata (numeric columns converted)."""
    section(f"Extra: budgets and revenues under ${LIMIT:,}")
    m = meta.drop_duplicates("id")
    cols = ["id", "title", "release_date", "budget", "revenue"]

    def table(df: pd.DataFrame, sort_by: str) -> None:
        df = df.sort_values(sort_by, ascending=False, na_position="last").head(ROWS)
        show([[r.id, short(r.title, 40),
               "" if r.release_date is None or str(r.release_date) == "NaT" else f"{r.release_date:%Y-%m-%d}",
               r.budget, r.revenue] for r in df[cols].itertuples()],
             headers=cols)

    small_budget = m[(m["budget"] > 0) & (m["budget"] < LIMIT)]
    print(f"\n{len(small_budget)} movies with 0 < budget < {LIMIT} "
          f"(of which budget == 1: {int((small_budget['budget'] == 1).sum())}). "
          f"Top {ROWS} by revenue:")
    table(small_budget, "revenue")

    small_revenue = m[(m["revenue"] > 0) & (m["revenue"] < LIMIT)]
    print(f"\n{len(small_revenue)} movies with 0 < revenue < {LIMIT} "
          f"(of which revenue == 1: {int((small_revenue['revenue'] == 1).sum())}). "
          f"Top {ROWS} by budget:")
    table(small_revenue, "budget")
