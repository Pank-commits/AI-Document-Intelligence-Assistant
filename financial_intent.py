import re

FINANCIAL_MAP = {
    "profit": [
        "net income", "profit", "earnings", "income", "made money",
        "profitability", "loss", "net profit"
    ],
    "revenue": [
        "revenue", "sales", "turnover", "total income", "top line"
    ],
    "expense": [
        "expenses", "operating expenses", "cost", "expenditure", "spending"
    ],
    "assets": [
        "total assets", "assets", "owned"
    ],
    "liabilities": [
        "total liabilities", "liabilities", "debt", "owed"
    ],
    "equity": [
        "shareholders equity", "equity", "net worth", "book value"
    ]
}

def detect_metric(query: str):
    q = query.lower()

    for key, aliases in FINANCIAL_MAP.items():
        for a in aliases:
            pattern = r"\b" + re.escape(a) + r"\b"
            if re.search(pattern, q):
                return key
    return None


def detect_years(query: str):
    years = re.findall(r"(20\d{2})", query)
    return [int(y) for y in years]