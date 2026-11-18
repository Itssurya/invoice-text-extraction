import logging
import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger("invoice-api")

AWS_REGION = (
    os.getenv("AWS_DEFAULT_REGION")
    or os.getenv("AWS_REGION")
    or "ap-south-1"
)

# Token that callers must send in the "x-api-token" header (set in .env).
API_TOKEN = os.getenv("API_TOKEN", "").strip()


def load_allowed_locations() -> list[tuple[str, str]]:
    """Buckets/folders the API may read, from ALLOWED_S3_LOCATIONS (comma separated), e.g.
    "ezship-track/eshipz/,invoice-pod-merge/invoice_parser/". Falls back to S3_BUCKET_NAME + S3_PREFIX."""
    raw = os.getenv("ALLOWED_S3_LOCATIONS", "").strip()
    if not raw:
        bucket = os.getenv("S3_BUCKET_NAME", "").strip()
        raw = f"{bucket}/{os.getenv('S3_PREFIX', '')}" if bucket else ""

    locations = []
    for item in raw.split(","):
        bucket, _, prefix = item.strip().removeprefix("s3://").partition("/")
        prefix = prefix.lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        if bucket:
            locations.append((bucket, prefix))
    return locations


ALLOWED_S3_LOCATIONS = load_allowed_locations()  # buckets must be in AWS_REGION

# Textract job settings
JOB_TIMEOUT_SECONDS = int(os.getenv("TEXTRACT_TIMEOUT_SECONDS", "120"))
POLL_INTERVAL_START = 1.0  # seconds; grows slowly up to POLL_INTERVAL_MAX
POLL_INTERVAL_MAX = 3.0
# Leave TEXTRACT_FEATURES empty to use plain text detection (faster and cheaper).
TEXTRACT_FEATURES = [
    f.strip().upper()
    for f in os.getenv("TEXTRACT_FEATURES", "LAYOUT,TABLES,FORMS").split(",")
    if f.strip()
]

# Textract errors caused by the file itself (reported as 400, not 502)
FILE_ERRORS = {
    "InvalidS3ObjectException": "Textract cannot read this S3 file. Check the key and that the bucket is in "
                                "the same region as AWS_REGION.",
    "UnsupportedDocumentException": "Unsupported document. Send a valid, unencrypted PDF.",
    "BadDocumentException": "The PDF is corrupted or cannot be read.",
    "DocumentTooLargeException": "The PDF is too large for Textract.",
}

_clients: dict[str, Any] = {}


def get_client(service: str):
    """boto3 client for "s3" or "textract", created on first use."""
    if service not in _clients:
        options: dict[str, Any] = {"region_name": AWS_REGION}
        key = os.getenv("AWS_ACCESS_KEY_ID")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        if key and secret:  # keys from .env; otherwise boto3 uses the server's IAM role
            options["aws_access_key_id"] = key
            options["aws_secret_access_key"] = secret
            token = os.getenv("AWS_SESSION_TOKEN")  # optional
            if token:
                options["aws_session_token"] = token
        _clients[service] = boto3.client(
            service,
            config=Config(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=5,
                read_timeout=60,
            ),
            **options,
        )
    return _clients[service]


app = FastAPI(title="Invoice PDF -> JSON (S3 + Amazon Textract)")


# --------------------------------------------------
# Request: S3 link -> (bucket, key)
# --------------------------------------------------

def parse_s3_location(value: str) -> tuple[str, str]:
    """Accepts s3://bucket/key or an https S3 object URL (path-style or virtual-hosted)."""
    parsed = urlparse(value.strip())

    if parsed.scheme == "s3":
        bucket, key = parsed.netloc, parsed.path.lstrip("/")
    elif parsed.scheme in ("http", "https") and (parsed.hostname or "").endswith(".amazonaws.com"):
        host = parsed.hostname.split(".")  # hostname has no port, e.g. ":443"
        path = unquote(parsed.path).lstrip("/")
        if host[0] == "s3" or host[0].startswith("s3-"):  # https://s3.<region>.amazonaws.com/<bucket>/<key>
            bucket, _, key = path.partition("/")
        else:  # https://<bucket>.s3.<region>.amazonaws.com/<key>
            bucket, key = host[0], path
    else:
        raise HTTPException(
            400,
            "s3_link must be an S3 object link, e.g. s3://<bucket>/<folder>/<file>.pdf",
        )

    if not bucket or not key:
        raise HTTPException(400, "s3_link must include both the bucket and the file path.")
    return bucket, key


def validate_location(bucket: str, key: str) -> None:
    if not ALLOWED_S3_LOCATIONS:
        raise HTTPException(500, "ALLOWED_S3_LOCATIONS (or S3_BUCKET_NAME) is not configured in .env.")
    if ".." in key.split("/") or not any(
        bucket == allowed_bucket and key.startswith(prefix)
        for allowed_bucket, prefix in ALLOWED_S3_LOCATIONS
    ):
        raise HTTPException(400, "This bucket or folder is not allowed.")
    if not key.lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files can be processed.")


# --------------------------------------------------
# Asynchronous Textract job on the PDF that is already in S3
# --------------------------------------------------

def start_job(textract, bucket: str, key: str) -> str:
    location = {"S3Object": {"Bucket": bucket, "Name": key}}
    if TEXTRACT_FEATURES:
        response = textract.start_document_analysis(
            DocumentLocation=location,
            FeatureTypes=TEXTRACT_FEATURES,
        )
    else:
        response = textract.start_document_text_detection(DocumentLocation=location)
    return response["JobId"]


def get_job_page(textract, job_id: str, token: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"JobId": job_id, "MaxResults": 1000}
    if token:
        kwargs["NextToken"] = token
    if TEXTRACT_FEATURES:
        return textract.get_document_analysis(**kwargs)
    return textract.get_document_text_detection(**kwargs)


def collect_blocks(textract, job_id: str) -> list[dict]:
    """Poll until the job finishes, then read every result page."""
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    delay = POLL_INTERVAL_START

    while True:
        time.sleep(delay)
        delay = min(delay * 1.5, POLL_INTERVAL_MAX)
        response = get_job_page(textract, job_id)
        status = response.get("JobStatus")

        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "PARTIAL_SUCCESS"):
            reason = response.get("StatusMessage") or "no details"
            logger.error("Textract job %s ended with %s: %s", job_id, status, reason)
            raise HTTPException(502, f"Textract could not read the whole PDF ({status}): {reason}")
        if time.monotonic() > deadline:
            raise HTTPException(504, f"Textract did not finish within {JOB_TIMEOUT_SECONDS} seconds.")

    blocks = list(response.get("Blocks", []))
    token = response.get("NextToken")
    while token:  # results come in chunks of up to 1000 blocks
        response = get_job_page(textract, job_id, token)
        blocks.extend(response.get("Blocks", []))
        token = response.get("NextToken")
    return blocks


def blocks_to_pages(blocks: list[dict]) -> list[dict[str, Any]]:
    by_page: dict[int, list[dict]] = {}
    for block in blocks:
        if block.get("BlockType") == "LINE" and block.get("Text"):
            confidence = block.get("Confidence")
            box = block.get("Geometry", {}).get("BoundingBox", {})
            by_page.setdefault(block.get("Page", 1), []).append({
                "text": block["Text"],
                "confidence": round(confidence, 2) if confidence is not None else None,
                "left": round(box.get("Left", 0.0), 4),
                "top": round(box.get("Top", 0.0), 4),
            })

    return [
        {
            "page": number,
            "line_count": len(lines),
            "lines": lines,
            "text": "\n".join(line["text"] for line in lines),
        }
        for number, lines in sorted(by_page.items())
    ]


def run_ocr(bucket: str, key: str) -> tuple[list[dict], str]:
    """Run one Textract job on s3://bucket/key. Blocking: use a threadpool."""
    s3 = get_client("s3")
    textract = get_client("textract")

    try:  # clear error for a wrong file name, before paying for a Textract job
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            raise HTTPException(404, f"File not found in S3: s3://{bucket}/{key}") from exc
        raise

    job_id = start_job(textract, bucket, key)
    blocks = collect_blocks(textract, job_id)

    pages = blocks_to_pages(blocks)
    if not pages:
        raise HTTPException(422, "No text was found in the PDF.")

    full_text = "\n\n".join(f"--- Page {p['page']} ---\n{p['text']}" for p in pages)
    return pages, full_text


NUM = r"[\d.,']+"

NOISE_EXACT = {
    "commercial invoice", "easyjet.com", "scomar",
    "item", "description", "qty", "total price",
}

COUNTRY_NAMES = {
    "ITALY": "IT", "UNITED KINGDOM": "GB", "GERMANY": "DE", "FRANCE": "FR",
    "SPAIN": "ES", "NETHERLANDS": "NL", "UNITED STATES": "US",
}

UK_POSTCODE = re.compile(r"^(?P<pc>[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\s+(?P<city>.+)$")
POSTAL_NUMERIC = re.compile(r"^(?P<pc>\d{4,6})\s+(?P<city>.+)$")

DUMMY_PHONE = "9876543210"


def to_float(value: str) -> float:
    return float(value.replace("'", "").replace(",", ""))


def clean_text(text: str) -> str:
    kept = []
    prev_was_time = False
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if not line:
            continue
        # the single uppercase code right after the page time (SCOMAR, AVINKU ...) is header noise
        if prev_was_time:
            prev_was_time = False
            if re.fullmatch(r"[A-Z]{3,12}", line):
                continue
        if re.match(r"^--- page \d+ ---$", low):
            continue
        if low in NOISE_EXACT:
            continue
        if low.startswith("produced by") or low in {"amos", "www.swiss-as.com"}:
            continue
        if re.match(r"^page\s*\d[\d\s/]*$", low):
            continue
        if re.match(r"^\d{1,2}:\d{2}$", line):
            prev_was_time = True
            continue
        # page-header date (e.g. 29.Sep.2026) repeats on every page and can split an item
        if re.fullmatch(r"\d{1,2}\.\s*[A-Za-z]{3}\.\s*\d{4}", line):
            continue
        kept.append(line)
    return "\n".join(kept)


def parse_date(text: str) -> str:
    m = re.search(r"(\d{1,2})\.\s*([A-Za-z]{3})\.\s*(\d{4})", text)
    if not m:
        return ""
    d, mon, y = m.groups()
    return datetime.strptime(f"{d} {mon.title()} {y}", "%d %b %Y").strftime("%d/%m/%Y")


def empty_party(party_type: str = "business") -> dict[str, Any]:
    return {
        "contact_name": "", "company_name": "", "street1": "", "street2": "",
        "city": "", "state": "", "postal_code": "", "phone": "", "email": "",
        "tax_id": "", "country": "", "type": party_type,
    }


def _is_country(line: str) -> bool:
    return line.upper() in COUNTRY_NAMES


def _match_postal(line: str):
    return UK_POSTCODE.match(line) or POSTAL_NUMERIC.match(line)


def _valid_column(col: list[str]) -> bool:
    # a real address column has exactly one postal/city line and one country line
    return (
        sum(_is_country(l) for l in col) == 1
        and sum(bool(_match_postal(l)) for l in col) == 1
    )


def split_columns(lines: list[str]) -> tuple[list[str], list[str]]:
    # OCR usually reads the two side-by-side address blocks row by row
    # (left, right, left, right ...). A column ends at its country line.
    cols: tuple[list[str], list[str]] = ([], [])
    done = [False, False]
    turn = 0
    for line in lines:
        c = turn if not done[turn] else 1 - turn
        cols[c].append(line)
        if _is_country(line):
            done[c] = True
        turn = 1 - c
    if all(_valid_column(c) for c in cols):
        return cols

    # fallback: OCR returned one whole column after the other
    idx = next((i for i, l in enumerate(lines) if _is_country(l)), None)
    if idx is not None:
        first, second = lines[: idx + 1], lines[idx + 1:]
        if _valid_column(first) and _valid_column(second):
            return first, second
    return cols


def parse_column(col: list[str]) -> dict[str, Any]:
    party = empty_party()
    text_lines = []
    province = ""
    for line in col:
        if _is_country(line):
            party["country"] = COUNTRY_NAMES[line.upper()]
        elif (m := _match_postal(line)):
            party["postal_code"] = m["pc"]
            city, _, state = m["city"].partition(",")  # "SLINFOLD, WEST SUSSEX"
            party["city"] = city.strip().title()
            party["state"] = state.strip().title()
        else:
            text_lines.append(line)
            prov = re.search(r"\(([A-Z]{2})\)\s*$", line)  # "SOMMA LOMBARDO (VA)"
            if prov:
                province = prov[1]
    if not party["state"]:
        party["state"] = province

    # order on the invoice: contact, company, street1, street2
    for key, value in zip(("contact_name", "company_name", "street1", "street2"), text_lines):
        party[key] = value
    if len(text_lines) > 4:
        party["street2"] = ", ".join(text_lines[3:])
    return party


def columns_from_geometry(lines: list[dict]) -> tuple[list[str], list[str]] | None:
    # Split the SHIP TO / SHIPPED BY block into two columns using each line's x/y position,
    # because the text order of the two columns changes from invoice to invoice.
    headers = [l for l in lines if l["text"].strip().upper().startswith(("SHIP TO", "SHIPPED BY"))]
    end = next((l for l in lines if l["text"].strip().lower().startswith("carrier")), None)
    if not headers or end is None:
        return None
    top_limit = max(h["top"] for h in headers) + 0.003
    block = [l for l in lines if top_limit < l["top"] < end["top"] - 0.003]
    if len(block) < 4:
        return None

    lefts = sorted({round(l["left"], 3) for l in block})
    gaps = [(lefts[i + 1] - lefts[i], i) for i in range(len(lefts) - 1)]
    if not gaps:
        return None
    gap, idx = max(gaps)
    if gap < 0.1:  # no clear two-column layout
        return None
    split = (lefts[idx] + lefts[idx + 1]) / 2

    cols: tuple[list[str], list[str]] = ([], [])
    for l in sorted(block, key=lambda x: x["top"]):
        cols[0 if l["left"] < split else 1].append(l["text"])
    return cols


def parse_parties(raw_text: str, page_lines: list[dict] | None = None) -> tuple[dict, dict, list[str]]:
    warnings: list[str] = []

    def tidy(lines: list[str]) -> list[str]:
        out = [l.strip().strip("* ").strip() for l in lines]
        return [l for l in out if l and "VAT NUMBER" not in l.upper() and "EORI NUMBER" not in l.upper()]

    vat = re.search(r"VAT NUMBER:\s*([A-Z]{2}\w+)", raw_text, re.I)
    eori = re.search(r"EORI NUMBER:\s*([A-Z]{2}\w+)", raw_text, re.I)
    vat_id = vat.group(1).upper() if vat else ""
    eori_id = eori.group(1).upper() if eori else ""

    col_a = col_b = None
    geo = columns_from_geometry(page_lines) if page_lines else None
    if geo:
        a_lines, b_lines = tidy(geo[0]), tidy(geo[1])
        if _valid_column(a_lines) and _valid_column(b_lines):
            col_a, col_b = a_lines, b_lines

    if col_a is None:
        m = re.search(r"SHIPPED BY\s*(.*?)\s*Carrier:", raw_text, re.S | re.I)
        lines = tidy(m.group(1).splitlines()) if m else []
        if not lines:
            warnings.append("Address block (SHIPPED BY ... Carrier:) not found")
        col_a, col_b = split_columns(lines)
        if page_lines:
            warnings.append("Address columns guessed from text order; verify ship_from/ship_to")

    party_a, party_b = parse_column(col_a), parse_column(col_b)

    # the party whose country matches the VAT prefix is the consignee (ship_to)
    vat_country = vat_id[:2]
    if vat_country and party_a["country"] == vat_country and party_b["country"] != vat_country:
        ship_to, ship_from = party_a, party_b
    else:
        ship_from, ship_to = party_a, party_b

    # tax ids go to the party of the same country (VAT -> ship_to, EORI -> ship_from)
    for tax_id in (vat_id, eori_id):
        if not tax_id:
            continue
        target = next((p for p in (ship_from, ship_to) if p["country"] == tax_id[:2]), None)
        if target is None and tax_id == vat_id:
            target = ship_to
        if target is not None and not target["tax_id"]:
            target["tax_id"] = tax_id

    for name, party in (("ship_from", ship_from), ("ship_to", ship_to)):
        if not party["postal_code"] or not party["country"]:
            warnings.append(f"{name}: postal code or country not found")
    return ship_from, ship_to, warnings


LABELS = (
    r"(?:S/N|B/N|Weight \(KG\)|Condition|Order No|Customs No|Comm Code|"
    r"Manufacturer|Country of Origin):"
)
DEFAULT_ITEM_WEIGHT = 0.1  # used when the invoice line has no weight


def normalize_origin(value: str) -> str:
    v = value.strip().upper()
    if v == "UK":
        return "GB"
    if re.fullmatch(r"[A-Z]{2}", v):
        return v
    return "<TBA_ON_INVOICE>"


def parse_item(chunk: str) -> dict[str, Any] | None:
    m = re.match(
        rf"\s*(?P<pn>\S+)\s+(?P<qty>{NUM})\s+(?P<cur>[A-Z]{{3}})\s+(?P<total>{NUM})\s+"
        rf"Description:\s*(?P<desc>.+?)\s+(?P<unit>\S+)\s+(?={LABELS})",
        chunk, re.S,
    )
    if not m:
        return None
    qty = to_float(m["qty"]) or 1.0
    total = to_float(m["total"])

    variant = re.search(
        r"(?:S/N|B/N):\s*(.+?)\s*(?=(?:Weight \(KG\)|Condition|Order No|Customs No|"
        r"Comm Code|Manufacturer|Country of Origin):|$)",
        chunk, re.S,
    )
    weight = re.search(rf"Weight \(KG\):\s*({NUM})", chunk)
    hs = re.search(r"Comm Code:\s*(\d+)", chunk)
    origin = re.search(r"Country of Origin:\s*(\S+)", chunk)
    return {
        "description": " ".join(m["desc"].split()),
        "origin_country": normalize_origin(origin[1]) if origin else "",
        "sku": m["pn"],
        "hs_code": hs[1] if hs else "",
        "variant": " ".join(variant[1].split()) if variant else "",
        "quantity": int(qty) if qty.is_integer() else qty,
        "price": {"amount": round(total / qty, 2), "currency": m["cur"]},
        "weight": {"value": to_float(weight[1]) if weight else DEFAULT_ITEM_WEIGHT, "unit": "kg"},
        "_line_total": total,
        "_weight_missing": weight is None,
    }


def parse_carrier(text: str) -> str:
    m = re.search(r"Carrier:[ \t]*(.*)", text)
    if not m:
        return ""
    value = m.group(1).strip()
    if not value:  # value may be on the next line
        rest = text[m.end():].lstrip("\n").splitlines()
        value = rest[0].strip() if rest else ""
    return "" if value.endswith(":") else value  # a following label means no carrier


def parse_invoice(ocr_text: str, pages: list[dict] | None = None) -> dict[str, Any]:
    raw_head = ocr_text
    text = clean_text(ocr_text)
    header, *blocks = re.split(r"Package No:", text)

    shipment_no = re.search(r"Shipment No:\s*(\d+)", header)
    carrier = parse_carrier(header)
    gross = re.search(rf"Total Gross Weight \(KG\):\s*({NUM})", header)
    expected_pkgs = re.search(r"Total Num\. Of Packages:\s*(\d+)", header)
    invoice_no = shipment_no[1] if shipment_no else ""

    parcels, warnings = [], []
    for idx, block in enumerate(blocks, start=1):
        dims = re.search(rf"L:\s*({NUM})\s*W:\s*({NUM})\s*H:\s*({NUM})", block)
        wt = re.search(rf"Gross Weight \(KG\):\s*({NUM})", block)
        items = []
        for chunk in re.split(r"P/N:", block)[1:]:
            item = parse_item(chunk)
            if item:
                if item.pop("_weight_missing"):
                    warnings.append(
                        f"Package {idx}: no weight for {item['sku']}, used {DEFAULT_ITEM_WEIGHT}"
                    )
                items.append(item)
            else:
                snippet = " ".join(chunk.split())[:60]
                warnings.append(f"Package {idx}: could not parse an item near '{snippet}'")
        if not items:
            continue
        names = ", ".join(dict.fromkeys(i["description"].title() for i in items))
        parcels.append({
            "description": f"Package {idx} - {names}",
            "box_type": "custom",
            "quantity": 1,
            "weight": {"value": to_float(wt[1]) if wt else 0.0, "unit": "kg"},
            "dimension": {
                "length": to_float(dims[1]) if dims else 0,
                "width": to_float(dims[2]) if dims else 0,
                "height": to_float(dims[3]) if dims else 0,
                "unit": "cm",
            },
            "items": items,
        })

    computed = round(sum(i["_line_total"] for p in parcels for i in p["items"]), 2)
    stated = re.search(rf"TOTAL PRICE:\s*({NUM})\s*([A-Z]{{3}})", text)
    currency = stated[2] if stated else "GBP"
    if stated and abs(to_float(stated[1]) - computed) > 0.01:
        warnings.append(f"Item total {computed} differs from invoice TOTAL PRICE {stated[1]}")
    if expected_pkgs and int(expected_pkgs[1]) != len(parcels):
        warnings.append(f"Invoice says {expected_pkgs[1]} packages, parsed {len(parcels)}")
    for p in parcels:
        for i in p["items"]:
            i.pop("_line_total")

    ship_from, ship_to, party_warnings = parse_parties(raw_head, pages[0]["lines"] if pages else None)
    warnings.extend(party_warnings)
    for party in (ship_from, ship_to):
        party["phone"] = party["phone"] or DUMMY_PHONE

    payload = {
        "description": carrier,
        "parcel_contents": "Aircraft spare parts",
        "charged_weight": {"unit": "KG", "value": to_float(gross[1]) if gross else 0.0},
        "invoice_number": invoice_no,
        "invoice_date": parse_date(raw_head),
        "collect_on_delivery": {"amount": 0, "currency": currency},
        "shipment": {
            "docket_number": invoice_no,
            "ship_from": ship_from,
            "ship_to": ship_to,
            "return_to": dict(ship_from),
            "parcels": parcels,
        },
    }
    return {"data": payload, "warnings": warnings}


def verify_token(x_api_token: str | None = Header(default=None)) -> None:
    """Checks the x-api-token header."""
    if not API_TOKEN:
        raise HTTPException(500, "API_TOKEN is not configured in .env.")
    if not x_api_token or not secrets.compare_digest(x_api_token.encode(), API_TOKEN.encode()):
        raise HTTPException(401, "Invalid or missing x-api-token header.")


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Invoice PDF -> JSON",
        "aws_region": AWS_REGION,
        "allowed_locations_configured": bool(ALLOWED_S3_LOCATIONS),
    }


@app.post("/extract-text", dependencies=[Depends(verify_token)])
async def extract_text(
    s3_link: str = Query(..., description="e.g. s3://my-bucket/folder/file.pdf or the https object URL"),
):
    """Header: x-api-token. Query: s3_link=<link to the PDF in S3>."""
    bucket, key = parse_s3_location(s3_link)
    validate_location(bucket, key)

    try:
        pages, full_text = await run_in_threadpool(run_ocr, bucket, key)
        parsed = parse_invoice(full_text, pages)

        for warning in parsed["warnings"]:
            logger.warning("Invoice parse warning: %s", warning)

        return {
            "status": "success",
            "data": parsed["data"],
        }
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        logger.exception("AWS call failed")
        if code in FILE_ERRORS:
            raise HTTPException(400, FILE_ERRORS[code]) from exc
        raise HTTPException(
            502,
            f"AWS call failed ({code}). Check credentials, IAM permissions (S3 and Textract), "
            "bucket name and region.",
        ) from exc
    except BotoCoreError as exc:
        logger.exception("AWS connection problem")
        raise HTTPException(502, "Could not connect to AWS. Check region and network access.") from exc