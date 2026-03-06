import pandas as pd
import re

# ---------------- METRIC DETECTION ----------------
METRIC_COLUMNS = {
    "revenue": ["revenue", "sales"],
    "profit": ["net income", "profit"],
    "assets": ["total assets"],
    "liabilities": ["total liabilities"],
    "equity": ["equity", "shareholders equity"],
    "expenses": ["expenses", "operating expenses"]
}

def find_metric(query, df):
    q = query.lower()

    for col in df.columns:
        col_l = col.lower()
        for metric, keywords in METRIC_COLUMNS.items():
            for k in keywords:
                if k in q and k in col_l:
                        return col, metric
    return None, None


def find_years(query):
    return [int(y) for y in re.findall(r"(20\d{2})", query)]

def detect_year_column(df):
    for col in df.columns:
        series = df[col]
        sample = series.astype(str).head(5).str.extract(r'(20\d{2})', expand=False)
        if sample.notna().sum() >= 2:
            return col
    # Prefer explicit year column name as fallback, then first column.
    for col in df.columns:
        if "year" in str(col).lower():
            return col
    return df.columns[0]

def to_number(x):
    """Robust financial numeric parser"""
    if pd.isna(x):
        return None

    x = str(x).strip()

    if x in ["", "-", "NA", "N/A", "null"]:
        return None

    # Handle accounting negative: (1234)
    negative = False
    if x.startswith("(") and x.endswith(")"):
        negative = True
        x = x[1:-1]

    # remove currency & commas
    x = x.replace(",", "").replace("₹", "").replace("$", "")

    multiplier = 1
    if x.endswith("B"):
        multiplier = 1_000_000_000
        x = x[:-1]
    elif x.endswith("M"):
        multiplier = 1_000_000
        x = x[:-1]
    elif x.endswith("K"):
        multiplier = 1_000
        x = x[:-1]

    try:
        value = float(x) * multiplier
        if negative:
            value = -value
        return value
    except:
        return None
    
# ---------------- MAIN ENGINE ----------------

def answer_tabular(query, df):
    year_col = detect_year_column(df)
    col, metric = find_metric(query, df)
    years = find_years(query)

    if col is None:
        return "Not available in the dataset"

    # 1) Single year lookup
    if len(years) == 1:
        year = years[0]
        row = df[df[year_col] == year]
        if row.empty:
            return f"No data available for {year}."
        value = to_number(row.iloc[0][col])
        if value is None:
            return "Value present but not numeric."
        return f"{metric.capitalize()} in {year} is {value}."

    # 2) Comparison
    if len(years) == 2:
        y1, y2 = years
        r1 = df[df[year_col]==y1][col]
        r2 = df[df[year_col]==y2][col]

        if r1.empty or r2.empty:
            return "Requested years not found."

        v1 = to_number(r1.iloc[0])
        v2 = to_number(r2.iloc[0])

        if v1 is None or v2 is None:
            return "Comparison not possible due to non numeric values."
        diff = v2 - v1

        if diff == 0:
            trend = "stayed the same"
            return f"{metric.capitalize()} {trend} from {y1} to {y2}."
        trend = "increased" if diff>0 else "decreased"

        return f"{metric.capitalize()} {trend} from {y1} to {y2} by {abs(diff)}."

    # 3) Trend

    series = df[col].apply(to_number).dropna()

    if series.empty:
        return "No numeric financial data found."
    
    if "trend" in query.lower() or "over time" in query.lower():
        change = series.iloc[-1] - series.iloc[0]
        direction = "increasing" if change>0 else "decreasing"
        return f"{metric.capitalize()} shows an overall {direction} trend."

    # 4) Aggregation
    if "total" in query.lower():
        return f"Total {metric} across all years is {series.sum():,.2f}."

    if "average" in query.lower():
        return f"Average {metric} is {series.mean():,.2f}."

    return "Not available in the dataset"
