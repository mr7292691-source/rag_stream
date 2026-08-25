# Master Prompt — Build Mortgage Document Extraction Multi-Agent System

Paste this whole prompt into Claude Code / Copilot Chat with your workspace open.
If you have the old repo available in the same workspace, keep it open too so the
assistant can reference it for schema migration (Step 3 below references this).

---

## PROMPT START

You are building a production multi-agent pipeline for mortgage document
processing, from zero, in this repository. Follow this spec exactly. Build
incrementally, in the phase order below, and show me each phase's output
before moving to the next.

### System purpose
Given a mortgage loan package (a scanned/digital file or set of files), the
system must:
1. **Classify** every page and produce a document manifest (which pages
   belong to which distinct document, and what type each document is).
2. **Extract** structured fields from each classified document, using a
   field schema specific to that document type.
3. **QC** every extracted field with a composite confidence score, and flag
   low-confidence fields in the output — no auto-retry.
4. **Aggregate** everything into one final structured JSON per loan
   package.

### Hard constraints
- Input formats: PDF, TIFF, PNG, JPEG. PDFs may be multi-page and may
  themselves contain multiple stitched documents.
- LLM backbone: Gemini 3.5, accessed exclusively through a **self-hosted
  LiteLLM proxy** (OpenAI-compatible endpoint) — never call Gemini's native
  SDK directly. Use the `litellm` Python package's `completion()` pointed
  at the proxy URL, with the proxy's configured model alias.
- Document type taxonomy (broad scope — not just loan application docs):
  `URLA_1003, W2, PAYSTUB, BANK_STATEMENT, TAX_RETURN, APPRAISAL,
  TITLE_COMMITMENT, TITLE_INSURANCE_POLICY, DEED, HAZARD_INSURANCE,
  FLOOD_CERT, CLOSING_DISCLOSURE, LOAN_ESTIMATE, PROMISSORY_NOTE,
  MORTGAGE_DEED_OF_TRUST, UNKNOWN`
- Low-confidence extractions are flagged in the output only. Do not build
  any retry, re-extraction, or self-correction loop.
- Orchestration: LangGraph, using a shared persistent state object across
  all nodes, with **conditional routing** from the classifier to the
  correct per-doc-type extractor (not a static linear chain — the set and
  order of extraction calls depends on what the classifier finds).

### Architecture to implement

```
Ingestion Agent → Classifier Agent (→ Document Manifest)
  → conditional fan-out to per-doc-type Extractor Agents
  → QC/Confidence Agent
  → Aggregator Agent → final output
```

- **Ingestion Agent**: normalizes all input formats to per-page rendered
  images; runs basic image-quality checks (resolution, skew, blank-page
  detection); assigns `page_id`s.
- **Classifier Agent**: batches rendered pages to Gemini (via LiteLLM
  proxy), detects document boundaries within the package, assigns each
  detected document a `doc_type` and `classification_confidence`, outputs
  the Document Manifest.
- **Extractor Agents**: one parameterized extractor (not N hand-written
  ones) that takes a `doc_type` + its field schema and returns field
  values with per-field `confidence` and `provenance`.
- **QC/Confidence Agent**: computes a weighted composite confidence per
  field from (a) model self-confidence, (b) rule-based format validation,
  (c) cross-document consistency checks against sibling documents in the
  same package, (d) source image quality from ingestion. Flags fields
  below a configurable threshold (thresholds vary by field criticality
  tier: critical / standard / low).
- **Aggregator Agent**: merges manifest + extractions + QC results into
  one final JSON per package.

### Build phases — execute in this order

**Phase 0 — Project scaffolding**
Set up the repo structure, `requirements.txt` (litellm, langgraph,
pydantic, pdf2image or pymupdf for PDF rendering, Pillow for image
handling, pytest), and `config.py` with LiteLLM proxy settings read from
environment variables (`LITELLM_PROXY_URL`, `LITELLM_PROXY_API_KEY`,
`GEMINI_MODEL_ALIAS`), confidence weights, and flagging thresholds.

**Phase 1 — Shared state**
Define `LoanPackageState` (TypedDict) with fields: `package_id`,
`input_path`, `pages`, `manifest`, `extractions` (merged dict keyed by
`doc_id`, with a reducer for parallel branches), `qc_results` (same),
`errors`, `final_output`.

**Phase 2 — Ingestion Agent**
Implement multi-format normalization (PDF page-split + render, TIFF
multi-frame split, PNG/JPEG passthrough) into per-page images, plus
quality flagging. Write unit tests with sample files of each format.

**Phase 3 — Classifier Agent**
Implement the LiteLLM proxy call with the classification system prompt
below, batching pages per call, parsing the JSON manifest response, and
handling documents that don't match any known type as `UNKNOWN` rather
than forcing a guess.

**Phase 4 — Schema migration**
Before writing extractors: check if an existing schema source is
available in this workspace (an older GenAI extraction repo/module). If
so, migrate its field schemas into `schemas/field_schemas.py` rather than
inventing new ones — preserve original field names, only remap doc_type
names to the taxonomy above where there's a clear match, and list
anything unmapped separately for review. If no existing schema source is
available, build starter schemas for at least `URLA_1003` and
`CLOSING_DISCLOSURE` and stub the rest.

**Phase 5 — Extractor Agent**
Implement one parameterized extractor function/node that takes `doc_type`
+ schema + page images, calls the LiteLLM proxy with the extraction system
prompt below, and returns fields with value/confidence/provenance per
field.

**Phase 6 — QC/Confidence Agent**
Implement the weighted composite scoring (weights in `config.py`):
model confidence, format/regex validation (deterministic, not LLM-based),
cross-document consistency checks, source image quality. Output
per-field `final_confidence`, `flagged`, and `reason`.

**Phase 7 — Aggregator Agent**
Merge manifest + extractions + qc_results into `final_output`.

**Phase 8 — LangGraph wiring**
Build the graph: sequential edges Ingestion → Classifier, then
**conditional edges** from Classifier fanning out to Extractor
invocations per `doc_id` in the manifest (parallel where the LangGraph
version supports it), converging into QC, then Aggregator. Add a CLI
entrypoint (`main.py --input <path>`) that runs a package through the
full graph and writes `final_output` to a JSON file.

**Phase 9 — Tests**
End-to-end test with at least one synthetic multi-document PDF covering
2–3 doc types, asserting the manifest, extraction, and QC stages all
produce non-empty, schema-valid output, and that a deliberately malformed
field (e.g. invalid SSN format) gets flagged with `reason:
format_validation_failed`.

### System prompts to use verbatim

**Classifier Agent:**
```
You are a document classification agent for mortgage loan packages.
You will be shown one or more consecutive page images from a scanned
loan file. Your job:

1. Identify document boundaries — a loan package often contains many
   distinct documents stitched together. Detect where one document
   ends and another begins.
2. Classify each identified document into exactly one of these types:
   [URLA_1003, W2, PAYSTUB, BANK_STATEMENT, TAX_RETURN, APPRAISAL,
    TITLE_COMMITMENT, TITLE_INSURANCE_POLICY, DEED, HAZARD_INSURANCE,
    FLOOD_CERT, CLOSING_DISCLOSURE, LOAN_ESTIMATE, PROMISSORY_NOTE,
    MORTGAGE_DEED_OF_TRUST, UNKNOWN]
3. For each document, report a classification_confidence between 0
   and 1, and the specific visual/textual cues that support the
   classification (e.g. form title, header, standard layout markers).

Respond ONLY with JSON matching this schema:
{
  "documents": [
    {
      "doc_id": string,
      "doc_type": string,
      "page_range": [start_page_id, end_page_id],
      "classification_confidence": float,
      "evidence": string
    }
  ]
}

If a document does not clearly match any known type, classify it as
UNKNOWN rather than guessing — do not force a best-fit label.
```

**Extractor Agent (parameterized by `{doc_type}` / `{field_schema_json}` / `{doc_id}`):**
```
You are a field extraction agent for a {doc_type} document.
You will be shown the page image(s) for a single document already
identified as this type. Extract the following fields:

{field_schema_json}

For EACH field, return:
- value: the extracted value (or null if not present/legible)
- confidence: your confidence in this specific extraction, 0-1
- provenance: brief description of where on the page you found it
  (e.g. "top-right box, labeled 'Loan Amount'")

Rules:
- Do not infer or calculate values that are not directly present on
  the document, unless the field schema explicitly asks for a
  computed field (in which case show your basis).
- If text is illegible, cut off, or ambiguous, set value to null and
  confidence low rather than guessing.
- Preserve exact formatting for identifiers (SSN, loan numbers,
  dates) as they appear.

Respond ONLY with JSON:
{
  "doc_id": "{doc_id}",
  "fields": {
    "<field_name>": {"value": ..., "confidence": float, "provenance": string}
  }
}
```

**QC/Confidence Agent:**
```
You are a quality-control agent reviewing extracted mortgage document
fields. You are given:
- The extracted fields with their self-reported confidence
- Validation results from rule-based checks (format/pattern pass-fail)
- Related field values from other documents in the same loan package
  (for cross-document consistency checks)

For each field, produce a final confidence assessment and flag status:
{
  "<field_name>": {
    "final_confidence": float,
    "flagged": bool,
    "reason": "low_model_confidence" | "format_validation_failed" |
              "cross_document_mismatch" | "missing_required_field" | "ok"
  }
}

A field should be flagged if final_confidence falls below the
package's configured threshold, OR if any rule-based/cross-document
check fails outright regardless of model confidence. Do not attempt
to correct or re-derive values — only assess and flag.
```

### Working rules for you (the coding assistant)
- Write real, runnable code — no pseudocode, no TODO stubs left in place
  of core logic (schema stubs for unmapped doc types are the one
  exception).
- Use `litellm.completion(model=<proxy alias>, api_base=<proxy url>,
  api_key=<proxy key>, messages=[...])` for every LLM call — centralize
  this in a single `llm_client.py` wrapper, don't scatter raw litellm
  calls across agent files.
- After each phase, run whatever tests exist and show me the output
  before proceeding to the next phase.
- If you hit a design ambiguity not covered by this spec, stop and ask
  rather than assuming.

## PROMPT END
