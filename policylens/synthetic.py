"""Generate a realistic *fictional* policy corpus for demos and tests.

The documents belong to an invented agency, the "National Public
Administration (NPA)". They mimic the structure of real federal issuances:
directives set policy, instructions implement it, and manuals give
procedures, and they cite each other and real external authorities.
Nothing here is an actual government policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

AGENCY = "National Public Administration (NPA)"

TOPICS = [
    {
        "series": 1000, "name": "Cybersecurity",
        "owner": "Chief Information Security Officer (CISO)",
        "authorities": ["44 U.S.C. 3551", "OMB M-22-09", "Executive Order 14028"],
        "subtopics": ["Risk Management Framework", "Incident Response", "Identity and Access Management",
                      "Vulnerability Management"],
        "policy": [
            "All NPA information systems must receive an authorization to operate before processing agency data.",
            "Multifactor authentication is required for every user account with access to agency networks.",
            "System owners shall categorize each system by the potential impact of a loss of confidentiality, integrity, or availability.",
            "Security incidents must be reported to the Security Operations Center (SOC) within one hour of discovery.",
            "Critical vulnerabilities shall be remediated within 15 calendar days of publication.",
            "Continuous monitoring data will be reviewed monthly by the Authorizing Official (AO).",
            "Privileged access must be granted on a least-privilege basis and reviewed every quarter.",
            "Zero trust architecture principles guide the design of all new network services.",
            "Encryption of data at rest and in transit is mandatory for systems handling sensitive information.",
            "Contractors operating systems on behalf of NPA must meet the same security requirements as agency staff.",
        ],
        "procedure": [
            "The system owner prepares a security plan and submits it to the Authorizing Official.",
            "The SOC triages each reported incident, assigns a severity level, and notifies affected offices.",
            "Vulnerability scans run weekly and results are uploaded to the agency dashboard.",
            "Account managers disable inactive accounts after 45 days without use.",
            "After an incident is contained, the response team documents lessons learned within 30 days.",
            "Access reviews are recorded and retained for audit purposes.",
        ],
    },
    {
        "series": 2000, "name": "Records Management",
        "owner": "Agency Records Officer (ARO)",
        "authorities": ["44 U.S.C. 3101", "36 CFR 1220", "OMB M-23-07"],
        "subtopics": ["Electronic Records", "Records Retention Schedules", "Email Management", "Records Disposition"],
        "policy": [
            "Every NPA employee is responsible for creating and maintaining records that document agency business.",
            "Federal records must be managed electronically in an approved recordkeeping system.",
            "Records may only be destroyed in accordance with an approved records retention schedule.",
            "Email of senior officials is retained permanently under the Capstone approach.",
            "Unauthorized removal or destruction of federal records must be reported to the Agency Records Officer.",
            "Records schedules shall be reviewed at least every five years to reflect changes in business processes.",
            "Departing employees must complete a records exit briefing before separation.",
            "Records in messaging applications must be captured into an official recordkeeping system within 20 days.",
            "Permanent records are transferred to the National Archives when they reach the end of their agency retention period.",
        ],
        "procedure": [
            "Records liaisons inventory the records held by their office each fiscal year.",
            "Disposition actions are documented on a records destruction certificate.",
            "The ARO coordinates proposed schedules with the Office of General Counsel (OGC).",
            "Offices transfer eligible permanent records using the electronic transfer process.",
            "Supervisors confirm that departing staff have transferred records to a successor custodian.",
        ],
    },
    {
        "series": 3000, "name": "Artificial Intelligence Governance",
        "owner": "Chief AI Officer (CAIO)",
        "authorities": ["OMB M-25-21", "Executive Order 13960", "Public Law 117-263"],
        "subtopics": ["AI Use Case Inventory", "High-Impact AI Risk Management", "Generative AI Use", "AI Workforce"],
        "policy": [
            "The Chief AI Officer coordinates the use of artificial intelligence across NPA.",
            "Each AI use case must be recorded in the agency AI use case inventory and updated annually.",
            "AI whose output serves as a principal basis for decisions with significant effects on rights or safety is treated as high-impact.",
            "High-impact AI requires pre-deployment testing, an impact assessment, and ongoing monitoring.",
            "Staff may use approved generative AI tools for drafting but remain accountable for the accuracy of any output.",
            "Sensitive or personally identifiable information must not be entered into unapproved generative AI services.",
            "An AI Governance Board reviews high-impact determinations and waivers.",
            "Individuals affected by AI-enabled decisions shall have access to timely human review and an appeal process.",
            "Model performance must be monitored for drift and for disparities across demographic groups.",
        ],
        "procedure": [
            "Program offices submit an AI intake form to the CAIO before acquiring or developing an AI capability.",
            "The CAIO screens each intake against the categories of AI presumed to be high-impact.",
            "Project teams document testing results and residual risks in the AI impact assessment.",
            "The AI Governance Board meets monthly and records its decisions in the governance register.",
            "Operators complete training on interpreting model outputs before receiving system access.",
        ],
    },
    {
        "series": 4000, "name": "Privacy Program",
        "owner": "Senior Agency Official for Privacy (SAOP)",
        "authorities": ["5 U.S.C. 552a", "Public Law 107-347", "OMB M-17-12"],
        "subtopics": ["Privacy Impact Assessments", "Breach Response", "System of Records Notices"],
        "policy": [
            "NPA collects personally identifiable information only when it is relevant and necessary to a mission purpose.",
            "A privacy impact assessment is required before developing or procuring any system that collects PII.",
            "Suspected breaches of PII must be reported within one hour to the Privacy Office.",
            "Systems of records require a published system of records notice before operation.",
            "Social Security numbers may only be collected where required by law.",
            "Privacy controls are assessed together with security controls during system authorization.",
            "Individuals have the right to request access to and amendment of their records.",
        ],
        "procedure": [
            "The program office drafts the privacy impact assessment with support from the Privacy Office.",
            "The Breach Response Team evaluates the risk of harm and determines whether notification is required.",
            "Privacy Act requests are logged and answered within statutory timeframes.",
            "The SAOP signs each completed privacy impact assessment before it is published.",
        ],
    },
    {
        "series": 5000, "name": "Acquisition",
        "owner": "Senior Procurement Executive (SPE)",
        "authorities": ["41 U.S.C. 1702", "48 CFR 1.101", "OMB M-25-22"],
        "subtopics": ["Acquisition Planning", "Source Selection", "Contract Administration", "Acquisition of AI"],
        "policy": [
            "Acquisition plans are required for all procurements above the simplified acquisition threshold.",
            "Source selection decisions must be documented and based only on the evaluation criteria in the solicitation.",
            "Contracting Officer Representatives (COR) must be certified before being appointed.",
            "Market research shall be conducted before issuing any solicitation.",
            "Contracts for AI systems must address data rights, model documentation, and vendor lock-in.",
            "Past performance information is recorded for every contract that exceeds the reporting threshold.",
            "Conflicts of interest must be disclosed by all members of a technical evaluation panel.",
        ],
        "procedure": [
            "The program office prepares requirements documents and an independent government cost estimate.",
            "The technical evaluation panel rates each proposal against the stated evaluation factors.",
            "The Contracting Officer (CO) negotiates terms and awards the contract.",
            "CORs review deliverables and certify invoices for payment.",
            "Contract files are closed out within the timeframes set by regulation.",
        ],
    },
    {
        "series": 6000, "name": "Telework and Workforce Flexibilities",
        "owner": "Chief Human Capital Officer (CHCO)",
        "authorities": ["Public Law 111-292", "5 U.S.C. 6502", "5 CFR 610"],
        "subtopics": ["Telework Eligibility", "Remote Work Agreements", "Alternative Work Schedules"],
        "policy": [
            "Employees in eligible positions may telework under a written telework agreement.",
            "Telework eligibility is based on the duties of the position, not on the individual employee.",
            "Remote work agreements change an employee's official duty station and require approval by the CHCO.",
            "Employees who telework must be available during core hours and protect government information at home.",
            "Supervisors may terminate a telework agreement for documented performance or conduct reasons.",
            "Telework does not change the terms and conditions of employment or pay entitlements beyond locality.",
            "During emergencies, telework-ready employees are expected to continue working from an alternate location.",
        ],
        "procedure": [
            "The employee completes telework training and submits a telework agreement to the supervisor.",
            "Supervisors review telework agreements at least annually.",
            "The Office of Human Capital reports telework participation data to the Office of Personnel Management.",
            "Requests for remote work are routed through the CHCO with a cost analysis.",
        ],
    },
    {
        "series": 7000, "name": "Emergency Management and Continuity",
        "owner": "Director of Emergency Management (DEM)",
        "authorities": ["42 U.S.C. 5121", "Executive Order 12656"],
        "subtopics": ["Continuity of Operations", "Occupant Emergency Plans", "Emergency Notification"],
        "policy": [
            "NPA maintains a continuity of operations plan to sustain essential functions during disruptions.",
            "Essential functions must be resumed within 12 hours of a continuity activation.",
            "Every NPA facility shall have an occupant emergency plan that is exercised annually.",
            "Employees must keep their emergency contact information current in the notification system.",
            "Continuity facilities are tested at least once a year.",
            "Orders of succession and delegations of authority are maintained for all key positions.",
        ],
        "procedure": [
            "The DEM activates the continuity plan on direction from the Administrator.",
            "Emergency notifications are sent by text, email, and voice within 15 minutes of a decision.",
            "Each office reviews its essential functions and updates them after every exercise.",
            "After-action reports are completed within 45 days of an exercise or real event.",
        ],
    },
    {
        "series": 8000, "name": "Data Management",
        "owner": "Chief Data Officer (CDO)",
        "authorities": ["Public Law 115-435", "44 U.S.C. 3520", "OMB M-19-23"],
        "subtopics": ["Data Governance", "Open Data", "Data Quality", "Enterprise Data Catalog"],
        "policy": [
            "NPA data is a strategic asset and shall be managed throughout its lifecycle.",
            "Every dataset must have a designated data steward and be registered in the enterprise data catalog.",
            "Public data assets are published in open, machine-readable formats unless an exemption applies.",
            "Data quality standards apply to all data used for decision making and AI model training.",
            "A Data Governance Body chaired by the Chief Data Officer sets agency data standards.",
            "Data sharing agreements are required before exchanging data with external organizations.",
            "Datasets used to train AI models must be documented, including their provenance and known limitations.",
        ],
        "procedure": [
            "Data stewards record metadata for each dataset in the catalog within 30 days of creation.",
            "The CDO publishes an annual open data plan.",
            "Data quality issues are logged and tracked to resolution by the responsible steward.",
            "Requests to share data externally are reviewed by the CDO, the Privacy Office, and the OGC.",
        ],
    },
]

GENERAL_ROLES = [
    "Office of General Counsel (OGC)", "Office of Inspector General (OIG)",
    "Chief Information Officer (CIO)", "Chief Financial Officer (CFO)",
]


def _issuance_id(kind: str, series: int, number: int) -> str:
    return f"{kind} {series}.{number:02d}"


def generate_policy_corpus(seed: int = 7) -> list[dict]:
    """Return a list of ``{"doc_id", "title", "text"}`` dicts (~40 documents)."""
    rng = np.random.default_rng(seed)
    docs = []
    directive_ids = {t["series"]: _issuance_id("Directive", t["series"], 1) for t in TOPICS}
    for t in TOPICS:
        series, name = t["series"], t["name"]
        d_id = directive_ids[series]
        instr_ids = [_issuance_id("Instruction", series, i + 2) for i in range(len(t["subtopics"]))]
        manual_id = _issuance_id("Manual", series, 10)
        owner_short = t["owner"].split("(")[-1].rstrip(")")
        related = [directive_ids[s] for s in rng.choice([x for x in directive_ids if x != series], 2, replace=False)]

        def body(kind, title, purpose, policy_n, proc_n, refs):
            lines = [f"NPA {kind} {title}", ""]
            lines += ["1. PURPOSE", "", purpose, ""]
            lines += ["2. APPLICABILITY", "",
                      f"This issuance applies to all organizational elements of the {AGENCY if kind.startswith('Directive') else 'NPA'}, "
                      "including employees, contractors, and detailees performing work on behalf of the agency.", ""]
            pol = list(rng.choice(t["policy"], size=min(policy_n, len(t["policy"])), replace=False))
            lines += ["3. POLICY", "", " ".join(pol), ""]
            roles = [t["owner"]] + list(rng.choice(GENERAL_ROLES, size=2, replace=False))
            resp = []
            for r in roles:
                short = r.split("(")[-1].rstrip(")")
                resp.append(f"The {r} " if r not in seen_defined else f"The {short} ")
                seen_defined.add(r)
                resp[-1] += {
                    t["owner"]: f"oversees implementation of this issuance and monitors compliance across NPA.",
                    "Office of General Counsel (OGC)": "provides legal advice on the interpretation of this issuance.",
                    "Office of Inspector General (OIG)": "may audit compliance with this issuance.",
                    "Chief Information Officer (CIO)": "ensures information technology supports the requirements of this issuance.",
                    "Chief Financial Officer (CFO)": "identifies resources needed to carry out this issuance.",
                }[r]
            lines += ["4. RESPONSIBILITIES", "", " ".join(resp), ""]
            if proc_n:
                proc = list(rng.choice(t["procedure"], size=min(proc_n, len(t["procedure"])), replace=False))
                lines += ["5. PROCEDURES", "", " ".join(proc), ""]
            lines += ["REFERENCES", ""] + [f"({chr(97 + i)}) {r}" for i, r in enumerate(refs)]
            return "\n".join(lines)

        seen_defined: set[str] = set()
        docs.append({
            "doc_id": d_id, "title": f"NPA {d_id}, {name}",
            "text": f"ID: {d_id}\n" + body(
                f"Directive {series}.01,", name,
                f"This Directive establishes policy and assigns responsibilities for the {name.lower()} program "
                f"of the {AGENCY} in accordance with {', '.join(t['authorities'])}. It is implemented by "
                f"{', '.join(instr_ids)} and {manual_id}.",
                6, 0, t["authorities"] + related),
        })
        for i_id, sub in zip(instr_ids, t["subtopics"]):
            seen_defined = set()
            refs = [d_id] + list(rng.choice(t["authorities"], size=1)) + \
                ([str(rng.choice([x for x in instr_ids if x != i_id]))] if rng.random() < 0.7 else [])
            docs.append({
                "doc_id": i_id, "title": f"NPA {i_id}, {sub}",
                "text": f"ID: {i_id}\n" + body(
                    f"{i_id},", sub,
                    f"This Instruction implements {d_id} by establishing procedures for {sub.lower()}. "
                    f"Questions about this Instruction should be directed to the {owner_short} office.",
                    4, 4, refs),
            })
        seen_defined = set()
        docs.append({
            "doc_id": manual_id, "title": f"NPA {manual_id}, {name} Procedures Manual",
            "text": f"ID: {manual_id}\n" + body(
                f"{manual_id},", f"{name} Procedures Manual",
                f"This Manual provides step-by-step procedures for carrying out {d_id} and the related "
                f"instructions {', '.join(instr_ids[:2])}.",
                3, 6, [d_id] + instr_ids[:2]),
        })
    return docs


def write_policy_corpus(folder: str | Path, seed: int = 7) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for d in generate_policy_corpus(seed):
        p = folder / (d["doc_id"].replace(" ", "_") + ".txt")
        p.write_text(d["text"], encoding="utf-8")
        paths.append(p)
    return paths
