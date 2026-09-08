import argparse
import os
import re
from typing import Iterable

import pandas as pd


COUNTRY_TERMS = [
    "india", "usa", "united states", "uk", "united kingdom", "china", "japan",
    "france", "germany", "canada", "australia", "brazil", "europe", "asia",
]
REGION_TERMS = ["north", "south", "east", "west", "central"]
METRIC_TERMS = [
    "sales", "revenue", "profit", "market share", "margin", "roi",
    "quantity", "units", "growth", "trend",
]
DIMENSION_TERMS = ["country", "region", "product", "category", "company", "market"]


def normalize_query(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"\s*#\d+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_type(query: str, existing: str) -> str:
    if str(existing or "").strip():
        return str(existing).strip()
    q = query.lower()
    if any(token in q for token in ["chart", "graph", "plot", "visualize", "dashboard"]):
        return "chart"
    if any(token in q for token in ["compare", "vs", "versus"]):
        return "comparison"
    if any(token in q for token in ["trend", "over time", "last quarter", "last month", "this year", "q1", "q2", "q3", "q4"]):
        return "time-based"
    if any(token in q for token in ["why", "explain", "red flags", "worried", "concerned"]):
        return "reasoning"
    return "structured"


def infer_difficulty(query: str, existing: str) -> str:
    if str(existing or "").strip():
        return str(existing).strip()
    q = query.lower()
    hard_patterns = [
        "high sales but low profit", "growing but not profitable",
        "increasing sales but decreasing profit", "highest growth over time",
        "what should i do next", "why is this happening",
    ]
    medium_patterns = [
        "compare", "trend", "break it down", "market share", "last quarter",
        "last month", "last year", "same for", "compare with",
    ]
    if any(pattern in q for pattern in hard_patterns):
        return "hard"
    if any(pattern in q for pattern in medium_patterns):
        return "medium"
    return "easy"


def infer_chart_flag(query: str, existing) -> str:
    current = str(existing or "").strip().lower()
    if current in {"yes", "no"}:
        return current
    q = query.lower()
    chart_tokens = [
        "chart", "graph", "plot", "visualize", "dashboard", "trend", "breakdown",
        "distribution", "compare", "comparison", "top 5",
    ]
    return "yes" if any(token in q for token in chart_tokens) else "no"


def infer_expected_behavior(query: str, query_type: str, chart_required: str) -> str:
    q = query.lower()
    if chart_required == "yes":
        return "chart"
    if any(token in q for token in ["same for", "compare with", "what about"]):
        return "follow-up"
    if query_type == "comparison":
        return "comparison table"
    if query_type in {"reasoning", "structured", "time-based"}:
        return "answer"
    return "answer"


def infer_expected_entities(query: str) -> str:
    q = query.lower()
    hits: list[str] = []
    for term in COUNTRY_TERMS + REGION_TERMS + METRIC_TERMS + DIMENSION_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", q):
            hits.append(term)
    return ", ".join(dict.fromkeys(hits))


def load_tabular_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def run_lightweight_eval(query_df: pd.DataFrame, dataset_path: str) -> pd.DataFrame:
    from tabular_engine import answer_tabular

    dataset = load_tabular_dataset(dataset_path)
    answers = []
    statuses = []
    for query in query_df["Query"].tolist():
        answer = answer_tabular(query, dataset)
        answers.append(answer)
        statuses.append("ok" if "not available in the dataset" not in str(answer).lower() else "unanswered")
    query_df["Eval_Answer"] = answers
    query_df["Eval_Status"] = statuses
    return query_df


def enrich_query_sheet(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "Query" not in result.columns:
        raise ValueError("The workbook must contain a 'Query' column.")

    result["Query"] = result["Query"].map(normalize_query)
    result["Type"] = [infer_type(q, existing) for q, existing in zip(result["Query"], result.get("Type", [""] * len(result)))]
    result["Difficulty"] = [infer_difficulty(q, existing) for q, existing in zip(result["Query"], result.get("Difficulty", [""] * len(result)))]
    result["Chart_Required"] = [infer_chart_flag(q, existing) for q, existing in zip(result["Query"], result.get("Chart_Required", [""] * len(result)))]
    result["Expected_Behavior"] = [infer_expected_behavior(q, t, c) for q, t, c in zip(result["Query"], result["Type"], result["Chart_Required"])]
    result["Expected_Entities"] = result["Query"].map(infer_expected_entities)
    result["Query_Duplicate_Count"] = result.groupby("Query")["Query"].transform("count")
    return result


def print_summary(df: pd.DataFrame) -> None:
    print(f"Rows: {len(df)}")
    print(f"Unique queries: {df['Query'].nunique()}")
    print("Type counts:")
    print(df["Type"].value_counts(dropna=False).to_string())
    print("Difficulty counts:")
    print(df["Difficulty"].value_counts(dropna=False).to_string())
    print("Chart flag counts:")
    print(df["Chart_Required"].value_counts(dropna=False).to_string())
    duplicates = df[df["Query_Duplicate_Count"] > 1]["Query"].nunique()
    print(f"Duplicate query groups: {duplicates}")


def write_excel(path: str, sheet_name: str, df: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich and optionally evaluate a finance query sheet.")
    parser.add_argument("--input", required=True, help="Path to the source Excel workbook.")
    parser.add_argument("--output", default=os.path.join("uploads", "rag_finance_training_dataset_enriched.xlsx"), help="Where to save the enriched workbook.")
    parser.add_argument("--sheet", default=None, help="Sheet name to load. Defaults to the first sheet.")
    parser.add_argument("--dataset", default=None, help="Optional CSV/XLSX dataset to run lightweight evaluations against.")
    args = parser.parse_args()

    workbook = pd.ExcelFile(args.input)
    sheet_name = args.sheet or workbook.sheet_names[0]
    df = workbook.parse(sheet_name)
    enriched = enrich_query_sheet(df)
    if args.dataset:
        enriched = run_lightweight_eval(enriched, args.dataset)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_excel(args.output, sheet_name, enriched)
    print_summary(enriched)
    print(f"Saved enriched workbook to: {os.path.abspath(args.output)}")
    if args.dataset:
        print(f"Evaluated against dataset: {os.path.abspath(args.dataset)}")


if __name__ == "__main__":
    main()
