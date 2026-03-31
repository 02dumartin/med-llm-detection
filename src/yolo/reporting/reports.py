from __future__ import annotations

import argparse
import html
import re
import zipfile
from pathlib import Path

import pandas as pd

from src.yolo.reporting.summary import (
    DEFAULT_RESULTS_ROOT,
    SUMMARY_COLUMNS,
    TEST_LABELS,
    apply_filters,
    collect_summary_rows,
)


PROJECT_ROOT = Path("/home/jovyan/aicon-gamma-datavol-1/hjgoh/med-llm-detection")
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"
LONG_COLUMNS = [
    "train_data",
    "model_name",
    "model_family",
    "variant",
    "test_data",
    "dataset_block",
    "conf",
    "iou",
    "class",
    *SUMMARY_COLUMNS,
    "source_dir",
]
BLOCK_COLUMNS = [
    "class",
    *SUMMARY_COLUMNS,
]
EVAL_NAME_RE = re.compile(r"^(?P<test_data>.+)_(?P<conf>[^_]+)_(?P<iou>[^_]+)$")
TEST_ORDER = ["fgart", "ddr", "eophtha", "idrid", "diaretdb1"]
CLASS_ORDER = ["MA", "HE", "EX", "SE", "lesion", "overall", "Total"]


def sanitize_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[\[\]\*:/\\?]+", "_", name).strip() or "Sheet"
    cleaned = cleaned[:31]
    candidate = cleaned
    idx = 1
    while candidate in used:
        suffix = f"_{idx}"
        candidate = f"{cleaned[:31 - len(suffix)]}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def build_long_df(
    results_root: Path,
    train_data: str | None,
    model_name: str | None,
    model_family: str | None,
    variant: str | None,
    conf: str | None,
    iou: str | None,
    tests: list[str] | None,
) -> pd.DataFrame:
    rows = collect_summary_rows(results_root)
    if not rows:
        raise SystemExit(f"No evaluation summary files found under: {results_root}")

    df = pd.DataFrame(rows)
    df = apply_filters(
        df,
        train_data=train_data,
        model_name=model_name,
        model_family=model_family,
        variant=variant,
        conf=conf,
        iou=iou,
        tests=tests,
    )
    if df.empty:
        raise SystemExit("No rows left after filtering.")

    test_order = {name: idx for idx, name in enumerate(TEST_ORDER)}
    class_order = {name: idx for idx, name in enumerate(CLASS_ORDER)}
    df["_test_order"] = df["test_data"].map(lambda x: test_order.get(x, 999))
    df["_class_order"] = df["class"].map(lambda x: class_order.get(x, 999))
    df = df.sort_values(
        by=["train_data", "model_family", "model_name", "variant", "_test_order", "_class_order", "class"],
        kind="stable",
    ).drop(columns=["_test_order", "_class_order"])
    return df


def collect_metrics_total_entries(results_root: Path) -> list[dict]:
    entries: list[dict] = []
    for summary_path in sorted(results_root.rglob("evaluation/summary.csv")):
        rel = summary_path.relative_to(results_root)
        if len(rel.parts) < 5:
            continue
        train_data = rel.parts[0]
        model_name = rel.parts[1]
        eval_name = rel.parts[2]
        match = EVAL_NAME_RE.match(eval_name)
        if not match:
            continue
        entries.append(
            {
                "train_data": train_data,
                "model_name": model_name,
                "test_data": match.group("test_data"),
                "conf": match.group("conf"),
                "iou": match.group("iou"),
                "summary_path": summary_path,
            }
        )

    if not entries:
        for metrics_total_path in sorted(results_root.rglob("metrics_total.csv")):
            rel = metrics_total_path.relative_to(results_root)
            if len(rel.parts) < 4:
                continue
            train_data = rel.parts[0]
            model_name = rel.parts[1]
            eval_name = rel.parts[-2]
            match = EVAL_NAME_RE.match(eval_name)
            if not match:
                continue
            entries.append(
                {
                    "train_data": train_data,
                    "model_name": model_name,
                    "test_data": match.group("test_data"),
                    "conf": match.group("conf"),
                    "iou": match.group("iou"),
                    "summary_path": metrics_total_path,
                }
            )
    return entries


def row_sort_key(class_name: str) -> tuple[int, str]:
    try:
        return (CLASS_ORDER.index(class_name), class_name)
    except ValueError:
        return (999, class_name)


def build_model_sheet_rows(entries: list[dict]) -> list[list]:
    rows: list[list] = []
    test_rank = {name: idx for idx, name in enumerate(TEST_ORDER)}
    entries = sorted(entries, key=lambda x: (test_rank.get(x["test_data"], 999), x["test_data"]))

    for entry in entries:
        df = pd.read_csv(entry["summary_path"])
        for col in BLOCK_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[BLOCK_COLUMNS]
        df = df.sort_values(by="class", key=lambda s: s.map(lambda x: row_sort_key(str(x))))

        title = f"{TEST_LABELS.get(entry['test_data'], entry['test_data'])} | conf={entry['conf']} | iou={entry['iou']}"
        rows.append([title])
        rows.append(BLOCK_COLUMNS.copy())
        rows.extend(df.values.tolist())
        rows.append([])

    return rows


def col_letter(idx: int) -> str:
    result = []
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        result.append(chr(65 + rem))
    return "".join(reversed(result))


def xml_cell(value, row_idx: int, col_idx: int) -> str:
    ref = f"{col_letter(col_idx)}{row_idx}"
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return f'<c r="{ref}"/>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = html.escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def worksheet_xml(rows: list[list]) -> str:
    max_cols = max((len(r) for r in rows), default=1)
    max_rows = max(len(rows), 1)
    dimension = f"A1:{col_letter(max_cols)}{max_rows}"
    row_xml = []
    for r_idx, row in enumerate(rows, start=1):
        cells = [xml_cell(value, r_idx, c_idx) for c_idx, value in enumerate(row, start=1)]
        row_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{dimension}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        '<sheetData>'
        + "".join(row_xml)
        + '</sheetData></worksheet>'
    )


def workbook_xml(sheet_names: list[str]) -> str:
    sheets = []
    for idx, name in enumerate(sheet_names, start=1):
        sheets.append(f'<sheet name="{html.escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets>' + "".join(sheets) + '</sheets></workbook>'
    )


def workbook_rels_xml(sheet_count: int) -> str:
    rels = []
    for idx in range(1, sheet_count + 1):
        rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + '</Relationships>'
    )


def root_rels_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )


def content_types_xml(sheet_count: int) -> str:
    overrides = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    for idx in range(1, sheet_count + 1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        + "".join(overrides)
        + '</Types>'
    )


def write_simple_xlsx(out_path: Path, sheets: list[tuple[str, list[list]]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml(len(sheets)))
        zf.writestr("_rels/.rels", root_rels_xml())
        zf.writestr("xl/workbook.xml", workbook_xml([name for name, _ in sheets]))
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml(len(sheets)))
        for idx, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", worksheet_xml(rows))


def create_master_workbook(entries: list[dict], out_path: Path) -> None:
    used: set[str] = set()
    sheets = []

    summary_rows = [["train_data", "model_name", "num_test_blocks", "conf_values", "iou_values"]]
    grouped: dict[tuple[str, str], list[dict]] = {}
    for entry in entries:
        grouped.setdefault((entry["train_data"], entry["model_name"]), []).append(entry)

    for (train_data, model_name), model_entries in sorted(grouped.items()):
        conf_values = ", ".join(sorted({str(e["conf"]) for e in model_entries}))
        iou_values = ", ".join(sorted({str(e["iou"]) for e in model_entries}))
        summary_rows.append([train_data, model_name, len(model_entries), conf_values, iou_values])
    sheets.append((sanitize_sheet_name("summary", used), summary_rows))

    for (train_data, model_name), model_entries in sorted(grouped.items()):
        sheet_rows = build_model_sheet_rows(model_entries)
        sheets.append((sanitize_sheet_name(f"{train_data}_{model_name}", used), sheet_rows))

    write_simple_xlsx(out_path, sheets)


def create_train_workbooks(results_root: Path, entries: list[dict], out_dir: Path) -> list[Path]:
    created: list[Path] = []
    grouped: dict[str, dict[str, list[dict]]] = {}
    for entry in entries:
        grouped.setdefault(entry["train_data"], {}).setdefault(entry["model_name"], []).append(entry)

    for train_data, model_map in sorted(grouped.items()):
        used: set[str] = set()
        sheets: list[tuple[str, list[list]]] = []

        summary_rows = [["model_name", "num_test_blocks", "conf_values", "iou_values"]]
        for model_name, model_entries in sorted(model_map.items()):
            conf_values = ", ".join(sorted({str(e["conf"]) for e in model_entries}))
            iou_values = ", ".join(sorted({str(e["iou"]) for e in model_entries}))
            summary_rows.append([model_name, len(model_entries), conf_values, iou_values])
        sheets.append((sanitize_sheet_name("summary", used), summary_rows))

        for model_name, model_entries in sorted(model_map.items()):
            sheet_rows = build_model_sheet_rows(model_entries)
            sheets.append((sanitize_sheet_name(model_name, used), sheet_rows))

        out_path = out_dir / f"{train_data}.xlsx"
        write_simple_xlsx(out_path, sheets)
        created.append(out_path)

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evaluation CSV/XLSX reports")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--train-data", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-family", type=str, default=None)
    parser.add_argument("--variant", choices=["4cls", "1cls"], default=None)
    parser.add_argument("--conf", type=str, default=None)
    parser.add_argument("--iou", type=str, default=None)
    parser.add_argument("--tests", nargs="*", default=None)
    args = parser.parse_args()

    df_long = build_long_df(
        results_root=args.results_root,
        train_data=args.train_data,
        model_name=args.model_name,
        model_family=args.model_family,
        variant=args.variant,
        conf=args.conf,
        iou=args.iou,
        tests=args.tests,
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    long_csv_path = args.reports_dir / "all_eval_summary.csv"
    df_long[LONG_COLUMNS].to_csv(long_csv_path, index=False)

    all_entries = collect_metrics_total_entries(args.results_root)
    filtered = apply_filters(
        df_long.copy(),
        train_data=args.train_data,
        model_name=args.model_name,
        model_family=args.model_family,
        variant=args.variant,
        conf=args.conf,
        iou=args.iou,
        tests=args.tests,
    )
    keep_keys = {
        (row.train_data, row.model_name, row.test_data, str(row.conf), str(row.iou))
        for row in filtered[["train_data", "model_name", "test_data", "conf", "iou"]].drop_duplicates().itertuples(index=False)
    }
    filtered_entries = [
        e for e in all_entries
        if (e["train_data"], e["model_name"], e["test_data"], str(e["conf"]), str(e["iou"])) in keep_keys
    ]

    master_xlsx_path = args.reports_dir / "master.xlsx"
    create_master_workbook(filtered_entries, master_xlsx_path)
    created = create_train_workbooks(args.results_root, filtered_entries, args.reports_dir)

    print(f"[DONE] long csv  -> {long_csv_path}")
    print(f"[DONE] master    -> {master_xlsx_path}")
    for path in created:
        print(f"[DONE] workbook  -> {path}")


if __name__ == "__main__":
    main()
