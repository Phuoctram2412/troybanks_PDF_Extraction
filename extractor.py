from pathlib import Path
from .bill_loader   import load_bill_pages
from .gemini_client import call_gemini, MODEL
from .validator     import validate_and_clean


def extract_bill(file_path: str,
                  model: str = MODEL,
                  dpi: int = 150) -> dict:
    """Extracts all billing fields from a PDF, JPG, or PNG bill."""
    print(f"\n{'='*55}")
    print(f" Gemini Bill Extraction")
    print(f" File  : {Path(file_path).name}")
    print(f" Model : {model}")
    print(f"{'='*55}")

    page_bytes           = load_bill_pages(file_path, dpi=dpi)
    print(f"  Sending {len(page_bytes)} page(s) to Gemini...")
    raw_result           = call_gemini(page_bytes, model=model)
    extracted, confidence = validate_and_clean(raw_result)

    fields_found = sum(1 for v in extracted.values() if v is not None)
    fields_total = len(extracted)
    low_conf     = [f for f, c in confidence.items() if c["level"] in ("LOW", "UNKNOWN")]

    print(f"\n  Extraction rate: {fields_found}/{fields_total} fields")
    if low_conf:
        print(f"  ⚠ Review: {', '.join(low_conf)}")

    return {
        "source_file":           Path(file_path).name,
        "model_used":            model,
        "extracted_fields":      extracted,
        "confidence":            confidence,
        "fields_found":          fields_found,
        "fields_total":          fields_total,
        "extraction_rate":       f"{fields_found / fields_total * 100:.0f}%",
        "low_confidence_fields": low_conf,
    }


def extract_bill_hybrid(file_path: str,
                         confidence_threshold: float = 0.70,
                         tier1_model: str = "gemini-2.5-flash",
                         tier2_model: str = "gemini-2.5-pro") -> dict:
    """Tries Flash first, escalates to Pro if confidence is low."""
    result         = extract_bill(file_path, model=tier1_model)
    low_conf_count = sum(
        1 for c in result["confidence"].values()
        if c["score"] is not None and c["score"] < confidence_threshold
    )
    if low_conf_count > result["fields_total"] * 0.25:
        print(f"\n  ↑ Escalating to {tier2_model}...")
        result             = extract_bill(file_path, model=tier2_model)
        result["escalated"] = True
    else:
        result["escalated"] = False
    return result


def print_extraction_report(result: dict):
    """Prints a human-readable extraction summary."""
    print(f"\n{'='*60}")
    print(f" EXTRACTION REPORT — {result['source_file']}")
    print(f" Model: {result['model_used']}  |  Rate: {result['extraction_rate']}")
    print(f"{'='*60}")
    print(f"{'Field':<28} {'Value':<25} {'Conf':<8} Status")
    print("─" * 72)

    for field, value in result["extracted_fields"].items():
        conf      = result["confidence"].get(field, {})
        score     = conf.get("score")
        level     = conf.get("level", "?")
        icon      = "✓" if level == "HIGH" else ("~" if level == "MEDIUM" else ("✗" if value is None else "⚠"))
        val_str   = str(value)[:23] if value else "null"
        score_str = f"{score:.2f}" if score is not None else "N/A"
        print(f"{field:<28} {val_str:<25} {score_str:<8} {icon} {level}")