import argparse
import io
import os
import time

import pandas as pd

import app as finance_app


def normalize_query(text: str) -> str:
    text = str(text or "").strip()
    return text


def infer_expected_route(query: str, declared_type: str = "") -> str:
    q = str(query or "").lower()
    declared = str(declared_type or "").lower().strip()
    if declared in {"chart", "comparison", "time-based", "aggregation", "retrieval", "reasoning"}:
        if declared in {"chart", "comparison", "time-based", "aggregation"}:
            return "structured"
        if declared in {"retrieval", "reasoning"}:
            return "semantic"
    if any(token in q for token in ["chart", "graph", "plot", "visualize", "dashboard"]):
        return "structured"
    if any(token in q for token in ["compare", "vs", "trend", "last quarter", "last month", "this year", "top 5", "highest", "lowest"]):
        return "structured"
    if any(token in q for token in ["why", "explain", "red flags", "concerned"]):
        return "semantic"
    return "unknown"


def wait_for_upload_ready(client, timeout_seconds: int = 180) -> dict:
    started = time.time()
    last_payload = {}
    while time.time() - started < timeout_seconds:
        response = client.get("/upload_status")
        last_payload = response.get_json(silent=True) or {}
        state = last_payload.get("state")
        if state == "ready":
            return last_payload
        if state == "failed":
            raise RuntimeError(f"Upload failed: {last_payload.get('error') or last_payload.get('message')}")
        time.sleep(1.0)
    raise TimeoutError(f"Upload did not become ready within {timeout_seconds}s. Last status: {last_payload}")


def upload_dataset(client, dataset_path: str) -> dict:
    with open(dataset_path, "rb") as handle:
        payload = {
            "files": (io.BytesIO(handle.read()), os.path.basename(dataset_path)),
        }
        response = client.post("/upload", data=payload, content_type="multipart/form-data")
    data = response.get_json(silent=True) or {}
    if response.status_code not in {200, 202}:
        raise RuntimeError(f"Upload failed with status {response.status_code}: {data}")
    return data


def classify_app_result(response_json: dict, query: str, declared_type: str = "", status_code: int | None = None) -> tuple[str, str]:
    if not isinstance(response_json, dict):
        return "invalid_response", "Response was not valid JSON."

    expected_route = infer_expected_route(query, declared_type)
    answer = str(response_json.get("answer", "") or "")
    answer_lower = answer.lower()
    chart_returned = bool(response_json.get("chart_data"))

    if status_code and status_code >= 500:
        if "connection error" in answer_lower or "internal error while processing query" in answer_lower:
            return "semantic_llm_failed", "The semantic/LLM path failed during query processing."
        return "server_error", f"HTTP {status_code} returned from /query."

    if status_code and status_code >= 400:
        return "request_error", f"HTTP {status_code} returned from /query."

    if "files are still being indexed" in answer_lower:
        return "upload_not_ready", "The dataset was still indexing when the query ran."
    if "no active session" in answer_lower:
        return "session_missing", "No active Flask session was available."
    if "not available in the dataset" in answer_lower:
        return "no_answer", "The app reported that the answer was not available in the dataset."
    if "internal error while processing query" in answer_lower:
        if expected_route == "semantic":
            return "semantic_llm_failed", "The semantic/LLM path failed during query processing."
        return "processing_failed", "The app hit an internal processing error."
    if chart_returned:
        if expected_route == "structured":
            return "chart_ok", "Structured/chart route returned chart data."
        return "chart_ok", "Chart data was returned."
    if answer.strip():
        if expected_route == "structured":
            return "structured_ok", "Structured route returned a textual answer."
        if expected_route == "semantic":
            return "semantic_ok", "Semantic route returned a textual answer."
        return "answer_ok", "The app returned a textual answer."
    return "empty_answer", "The app returned an empty answer."


def evaluate_queries(client, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    answers = []
    statuses = []
    reasons = []
    confidences = []
    chart_flags = []
    http_codes = []

    declared_types = result["Type"].tolist() if "Type" in result.columns else [""] * len(result)
    for query, declared_type in zip(result["Query"].tolist(), declared_types):
        response = client.post("/query", json={"query": normalize_query(query)})
        payload = response.get_json(silent=True) or {"answer": response.get_data(as_text=True)}
        status, reason = classify_app_result(payload, query, declared_type, response.status_code)
        answers.append(payload.get("answer"))
        statuses.append(status)
        reasons.append(reason)
        confidences.append(payload.get("confidence"))
        chart_flags.append("yes" if payload.get("chart_data") else "no")
        http_codes.append(response.status_code)

    result["App_Answer"] = answers
    result["App_Status"] = statuses
    result["App_Status_Reason"] = reasons
    result["App_Confidence"] = confidences
    result["App_Chart_Returned"] = chart_flags
    result["App_HTTP_Status"] = http_codes
    return result


def write_excel(path: str, sheet_name: str, df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)


def print_summary(df: pd.DataFrame) -> None:
    print(f"Rows evaluated: {len(df)}")
    if "App_Status" in df.columns:
        print("App status counts:")
        print(df["App_Status"].value_counts(dropna=False).to_string())
    if "App_Chart_Returned" in df.columns:
        print("Chart returned counts:")
        print(df["App_Chart_Returned"].value_counts(dropna=False).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the actual Flask app query route with an Excel query sheet.")
    parser.add_argument("--input", required=True, help="Path to the Excel workbook containing a Query column.")
    parser.add_argument("--dataset", required=True, help="Dataset file to upload to the Flask app session.")
    parser.add_argument("--output", default=os.path.join("uploads", "rag_finance_training_dataset_flask_eval.xlsx"), help="Output workbook path.")
    parser.add_argument("--sheet", default=None, help="Sheet name to load. Defaults to the first sheet.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of queries to evaluate.")
    parser.add_argument("--upload-timeout", type=int, default=180, help="Seconds to wait for upload indexing.")
    args = parser.parse_args()

    workbook = pd.ExcelFile(args.input)
    sheet_name = args.sheet or workbook.sheet_names[0]
    df = workbook.parse(sheet_name)
    if "Query" not in df.columns:
        raise ValueError("The workbook must contain a 'Query' column.")
    if args.limit:
        df = df.head(args.limit).copy()

    finance_app.app.config["TESTING"] = True
    with finance_app.app.test_client() as client:
        upload_result = upload_dataset(client, args.dataset)
        print(f"Upload accepted: {upload_result}")
        ready_status = wait_for_upload_ready(client, timeout_seconds=args.upload_timeout)
        print(f"Upload ready: {ready_status}")
        evaluated = evaluate_queries(client, df)

    write_excel(args.output, sheet_name, evaluated)
    print_summary(evaluated)
    print(f"Saved Flask evaluation workbook to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
