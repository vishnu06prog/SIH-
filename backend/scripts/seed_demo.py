"""Generate a realistic KMRL sample library and push it through the API.

Usage:
    python scripts/seed_demo.py                # against http://127.0.0.1:8000
    python scripts/seed_demo.py --url http://host:8000 --keep

The generated files are written to ``samples/`` so they can also be uploaded by
hand through the UI during a demo.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import httpx

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

SAMPLES_DIR = BACKEND_ROOT / "samples"


# --------------------------------------------------------------- file builders
def build_pdf(lines: list[str]) -> bytes:
    """A minimal, valid single-page PDF carrying a real text layer."""
    wrapped: list[str] = []
    for line in lines:
        while len(line) > 95:
            cut = line.rfind(" ", 0, 95)
            cut = cut if cut > 40 else 95
            wrapped.append(line[:cut])
            line = line[cut:].lstrip()
        wrapped.append(line)

    escaped = [
        line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        for line in wrapped
    ]
    body = "\n".join(f"({line}) Tj T*" for line in escaped)
    content = f"BT /F1 10 Tf 45 760 Td 14 TL\n{body}\nET".encode("latin-1", "replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for offset in offsets:
        out.write(f"{offset:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


def build_docx(paragraphs: list[str]) -> bytes:
    from docx import Document as DocxDocument

    document = DocxDocument()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def build_scan(lines: list[str]) -> bytes:
    """A 'scanned notice' image so the OCR path is exercised in the demo."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1000, 60 + 34 * len(lines)
    image = Image.new("RGB", (width, height), "#f7f6f2")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:  # pragma: no cover - font-dependent
        font = ImageFont.load_default()
    for index, line in enumerate(lines):
        draw.text((40, 30 + index * 34), line, fill="#1a1a1a", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ------------------------------------------------------------------- documents
DOCUMENTS: list[tuple[str, bytes]] = []


def pdf(name: str, lines: list[str]) -> None:
    DOCUMENTS.append((name, build_pdf(lines)))


def docx(name: str, paragraphs: list[str]) -> None:
    DOCUMENTS.append((name, build_docx(paragraphs)))


def txt(name: str, text: str) -> None:
    DOCUMENTS.append((name, text.encode("utf-8")))


def png(name: str, lines: list[str]) -> None:
    DOCUMENTS.append((name, build_scan(lines)))


pdf(
    "Safety_Circular_21_2025_Platform_Screen_Doors.pdf",
    [
        "KOCHI METRO RAIL LIMITED",
        "SAFETY CIRCULAR NO. 21/2025                                   Date: 12/08/2025",
        "",
        "Subject: Mandatory inspection of platform screen doors - Aluva to Maharaja's College.",
        "",
        "1. All station controllers shall complete a physical inspection of platform screen door",
        "   interlocks at their station and record the result in the shift register.",
        "2. The inspection is safety critical and must be completed by 30/09/2025.",
        "3. The Safety department is required to submit a consolidated compliance report to the",
        "   Commissioner of Metro Rail Safety within 15 days of completion.",
        "4. Any hazard or near miss observed during the inspection shall be reported immediately",
        "   to the Operations Control Centre.",
        "5. Failure to comply may attract an audit non-conformity and a statutory penalty.",
        "",
        "Rolling Stock is requested to make trainsets available during non-revenue hours for the",
        "door interface checks. Electrical shall arrange platform lighting for night inspections.",
        "",
        "Chief Safety Officer, KMRL",
    ],
)

pdf(
    "Maximo_Job_Card_JC-2291_Bogie_Overhaul.pdf",
    [
        "KMRL MAINTENANCE JOB CARD - MAXIMO EXPORT",
        "Job Card No: JC-2291            Trainset: TS-14            Depot: Muttom",
        "",
        "Work order: Preventive maintenance of bogie assembly and brake calliper replacement.",
        "Scheduled window: 18/09/2025 to 22/09/2025 during non-revenue hours.",
        "",
        "Tasks:",
        "- Inspect bogie frame welds and record any defect in the maintenance schedule.",
        "- Replace worn brake pads on car 1 and car 3; traction motor bearing greasing.",
        "- Pantograph carbon strip wear measurement; log readings in Maximo.",
        "- Wheel profile measurement; re-profiling required if flange wear exceeds limits.",
        "",
        "Spare parts requirement has been raised with Procurement against purchase order PO-4471.",
        "The rolling stock engineer must certify the trainset before it returns to revenue service.",
        "",
        "Prepared by: Section Engineer, Rolling Stock",
    ],
)

pdf(
    "Invoice_INV-8842_Metro_Spares_Pvt_Ltd.pdf",
    [
        "TAX INVOICE",
        "Metro Spares Pvt Ltd, Kalamassery, Ernakulam        GSTIN: 32AABCM1234K1Z5",
        "",
        "Invoice No: INV-8842                              Invoice Date: 05/09/2025",
        "Buyer: Kochi Metro Rail Limited, Finance Department",
        "Against Purchase Order: PO-4471 dated 14/08/2025",
        "",
        "Description                        Qty      Rate         Amount",
        "Brake pad assembly (BG-220)         40    6,250.00    2,50,000.00",
        "Traction motor bearing kit          12   12,500.00    1,50,000.00",
        "Pantograph carbon strip             24    2,083.33      50,000.00",
        "",
        "Taxable value 4,50,000.00   CGST 9% 40,500.00   SGST 9% 40,500.00",
        "Amount payable: Rs 5,31,000.00",
        "",
        "Payment is due within 30 days of receipt. Delay attracts interest as per the contract.",
        "Finance is requested to process the payment and confirm the accounts entry.",
    ],
)

docx(
    "Tender_Notice_NIT-2025-31_Depot_Expansion.docx",
    [
        "KOCHI METRO RAIL LIMITED - NOTICE INVITING TENDER (NIT-2025-31)",
        "Subject: Construction of a second stabling line and workshop bay at the new depot.",
        "Scope of work: civil works including foundation, track slab, drainage and workshop building.",
        "Estimated value: Rs 42.50 crore. Earnest money deposit: Rs 42,50,000.",
        "Bid submission closes on 15/10/2025 at 15:00 hrs. Technical bids open on 16/10/2025.",
        "Vendors must submit a quotation along with the EMD and prior experience certificates.",
        "Procurement shall publish any corrigendum on the KMRL e-procurement portal.",
        "The Projects department is responsible for the technical evaluation of the bids.",
        "Contract award is expected within 45 days of the technical evaluation report.",
    ],
)

pdf(
    "CMRS_Directive_Annual_Safety_Audit_2025.pdf",
    [
        "COMMISSIONER OF METRO RAIL SAFETY - SOUTHERN CIRCLE",
        "Directive No. CMRS/KMRL/2025/117                    Date: 01/09/2025",
        "",
        "Subject: Annual statutory safety audit and closure of previous non-conformities.",
        "",
        "1. KMRL shall submit the annual safety audit compliance statement by 31/10/2025.",
        "2. All open non-conformities from the 2024 audit must be closed within 30 days.",
        "3. Evacuation drills at every station shall be conducted and documented before the audit.",
        "4. The Ministry of Housing and Urban Affairs is to be kept informed of the closure status.",
        "5. Non-compliance with this regulatory directive is a serious statutory lapse and may",
        "   result in restrictions on revenue service.",
        "",
        "Safety, Operations and Rolling Stock are jointly responsible for the corrective actions.",
    ],
)

docx(
    "HR_Circular_Refresher_Training_Station_Staff.docx",
    [
        "KMRL HUMAN RESOURCES - CIRCULAR 44/2025",
        "Subject: Refresher training programme for station controllers and customer relations staff.",
        "All station staff must complete the two-day refresher training before 20/10/2025.",
        "The programme covers the revised emergency response protocol and the new safety bulletin.",
        "Attendance is mandatory; the training calendar is published on the HR SharePoint site.",
        "Employees on leave during the scheduled batch shall be accommodated in the next batch.",
        "Payroll will not process the quarterly incentive for staff who have not completed training.",
        "HR is responsible for coordinating with the Safety department on the course content.",
    ],
)

pdf(
    "Incident_Report_OCC_Signal_Failure_Edappally.pdf",
    [
        "KMRL OPERATIONS - INCIDENT REPORT",
        "Incident ID: INC-2025-206         Date of occurrence: 08/09/2025    Time: 08:42 hrs",
        "Location: Edappally station, up line",
        "",
        "Description: An axle counter failure caused the interlocking to withhold the signal for",
        "train service 1042. The train was held for 6 minutes and headway increased to 11 minutes",
        "during the morning peak. There was no injury and no damage to rolling stock.",
        "",
        "Immediate action: The Operations Control Centre applied degraded mode working and the",
        "Signalling and Telecom team reset the axle counter section at 08:51 hrs.",
        "",
        "Root cause: intermittent connection in the trackside junction box.",
        "Corrective action: S&T shall replace the junction box connectors by 25/09/2025 and",
        "review all axle counter sections on the corridor. Safety must be informed of closure.",
    ],
)

pdf(
    "Track_Inspection_Report_Q3_Aluva_Corridor.pdf",
    [
        "KMRL CIVIL ENGINEERING - QUARTERLY TRACK INSPECTION REPORT (Q3 2025)",
        "Section: Aluva to Kalamassery viaduct",
        "",
        "Track geometry recording was carried out on 02/09/2025 using the track recording trolley.",
        "Two alignment defects were recorded at chainage 12.480 and 14.220 exceeding the",
        "maintenance limit. Tamping is required before the next quarterly run.",
        "Rail weld inspection found no crack indications. Ballast profile is within tolerance.",
        "Bridge bearing inspection on the viaduct girder is due and has been scheduled.",
        "",
        "Recommendation: Civil works team shall complete tamping at both locations by 05/10/2025",
        "and submit the post-tamping recording to the Chief Engineer.",
    ],
)

txt(
    "Signalling_CBTC_Upgrade_Note.txt",
    """KMRL SIGNALLING & TELECOM - TECHNICAL NOTE

Subject: CBTC software upgrade and SCADA interface validation.

The signalling supplier has released version 4.2 of the CBTC application. The upgrade
covers ATP and ATO behaviour at reduced headway and a fix for the radio handover delay
observed between Kaloor and Town Hall.

Planned activity:
- Regression testing on the test track between 21/09/2025 and 24/09/2025.
- SCADA interface validation with the Operations Control Centre.
- Fibre optic link redundancy check across all interlocking huts.

Operations must be informed before any change to the revenue service configuration.
The IT department shall confirm that the new interface complies with the unified
namespace data model used for condition monitoring.
""",
)

pdf(
    "Electrical_OHE_Shutdown_Notice_Muttom.pdf",
    [
        "KMRL ELECTRICAL DEPARTMENT - POWER BLOCK NOTICE",
        "Notice No: ELE/2025/88                         Date: 10/09/2025",
        "",
        "Subject: Traction power shutdown for 33kV substation maintenance at Muttom.",
        "",
        "An overhead equipment shutdown is scheduled from 01:00 hrs to 04:30 hrs on 26/09/2025.",
        "The transformer oil filtration and breaker servicing will be carried out during the block.",
        "Earthing continuity checks on the OHE mast sections are included in the scope.",
        "",
        "Operations shall ensure no stabling movement is planned in the affected section.",
        "Rolling Stock must complete shunting before 00:45 hrs. Safety will supervise the",
        "permit-to-work issue. Station lighting will run on the UPS supply during the block.",
    ],
)

txt(
    "Bilingual_Station_Notice_Aluva.txt",
    """KOCHI METRO RAIL LIMITED - STATION NOTICE / സ്റ്റേഷൻ അറിയിപ്പ്

English:
Escalator number 3 at Aluva station will be under maintenance from 22/09/2025 to
24/09/2025. Passengers are requested to use the lift or the staircase. Station
controllers must display the notice at both concourse entries and inform the
Operations Control Centre if crowding is observed during peak hours.

മലയാളം:
അളുവ സ്റ്റേഷനിലെ മൂന്നാം നമ്പർ എസ്കലേറ്റർ 22/09/2025 മുതൽ 24/09/2025 വരെ
അറ്റകുറ്റപ്പണിയിലായിരിക്കും. യാത്രക്കാർ ലിഫ്റ്റ് അല്ലെങ്കിൽ പടിക്കെട്ട്
ഉപയോഗിക്കണമെന്ന് അഭ്യർത്ഥിക്കുന്നു. തിരക്ക് കൂടുതലുള്ള സമയങ്ങളിൽ സ്റ്റേഷൻ
കൺട്രോളർ ഓപ്പറേഷൻസ് കൺട്രോൾ സെന്ററിനെ അറിയിക്കണം.
""",
)

png(
    "Scanned_Notice_Depot_Entry_Control.png",
    [
        "KOCHI METRO RAIL LIMITED - DEPOT NOTICE",
        "",
        "Subject: Revised entry control at Muttom depot gate 2.",
        "",
        "All contractor staff must carry the new photo identity card from",
        "01/10/2025. Security shall verify the gate pass against the",
        "approved contractor list before allowing entry to the workshop.",
        "",
        "Safety induction is mandatory for every new contractor worker.",
    ],
)

pdf(
    "IT_Note_Unified_Namespace_IoT_Condition_Monitoring.pdf",
    [
        "KMRL INFORMATION TECHNOLOGY - ARCHITECTURE NOTE",
        "",
        "Subject: Unified Namespace (UNS) data streams and IoT condition monitoring rollout.",
        "",
        "The proposed architecture publishes rolling stock and OHE telemetry to a broker using a",
        "unified namespace hierarchy. Condition monitoring data from IoT sensors will feed the",
        "analytics dashboard used by the maintenance planning team.",
        "",
        "Scope: message broker deployment, database sizing, API integration with Maximo, and a",
        "machine learning model for early fault prediction on traction motors.",
        "The software licences must be procured before the pilot begins on 01/11/2025.",
        "Cyber security review of the network segmentation is required before go-live.",
    ],
)

docx(
    "Legal_Opinion_Contract_Variation_Depot_Works.docx",
    [
        "KMRL LEGAL DEPARTMENT - OPINION NOTE",
        "Subject: Contract variation claim raised by the depot works contractor.",
        "The contractor has raised a claim for additional payment citing a change in the scope of work.",
        "In our legal opinion the variation clause of the agreement permits the claim only where the",
        "engineer has issued a written instruction; no such instruction is on record.",
        "We advise that the claim be rejected and the contractor be put on notice within 21 days.",
        "If the contractor proceeds to arbitration, Projects and Finance must preserve all records.",
        "There is a litigation risk if the rejection is not communicated within the contractual period.",
    ],
)

pdf(
    "Board_Minutes_Meeting_112_Extracts.pdf",
    [
        "KOCHI METRO RAIL LIMITED - MINUTES OF THE 112th BOARD MEETING (EXTRACT)",
        "Date: 28/08/2025",
        "",
        "Agenda item 4: Corridor extension and the two new depots.",
        "Resolved that the detailed project report for the corridor extension be placed before the",
        "board in the next meeting. The Projects department shall present the milestone plan.",
        "",
        "Agenda item 5: Document management.",
        "The board noted the volume of documents received across departments and resolved that a",
        "document intelligence platform be implemented to reduce information latency and improve",
        "compliance tracking. IT is responsible for the implementation roadmap.",
        "",
        "Agenda item 6: Ridership and revenue.",
        "Operations reported an improvement in punctuality and average ridership for the quarter.",
    ],
)


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the KMRL demo library")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Backend base URL")
    parser.add_argument("--files-only", action="store_true", help="Only write samples to disk")
    args = parser.parse_args()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in DOCUMENTS:
        (SAMPLES_DIR / name).write_bytes(data)
    print(f"Wrote {len(DOCUMENTS)} sample files to {SAMPLES_DIR}")

    if args.files_only:
        return 0

    uploaded = 0
    with httpx.Client(base_url=args.url, timeout=120.0, trust_env=False) as client:
        try:
            client.get("/api/health").raise_for_status()
        except Exception as exc:
            print(f"Backend not reachable at {args.url}: {exc}")
            return 1

        for name, data in DOCUMENTS:
            response = client.post(
                "/api/documents/upload",
                files={"file": (name, data, "application/octet-stream")},
                data={"uploaded_by": "Demo Seeder"},
            )
            if response.status_code == 201:
                uploaded += 1
                print(f"  uploaded  {name}")
            else:
                print(f"  FAILED    {name}: {response.status_code} {response.text[:160]}")

        print(f"\nUploaded {uploaded}/{len(DOCUMENTS)} documents. Waiting for analysis…")
        for _ in range(60):
            stats = client.get("/api/documents/stats").json()
            if stats["in_progress"] == 0:
                break
            time.sleep(1)
        stats = client.get("/api/documents/stats").json()
        print(
            f"Library: {stats['total_documents']} documents, "
            f"{stats['analysis_complete']} analysed, {stats['failed']} failed, "
            f"{stats['open_actions']} action items."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
