# Zindian Orchestrator Documentation Structure Map (v2.5)

This document provides a consolidated, unified map of the structure, cross-document overlaps, and ownership details for all nine files in the Zindian Orchestrator documentation suite. This map helps developers navigate the documentation and maintain structure during the v2.5 lean restructure.

---

## 1. Document Structure Map

### source_of_truth.md (v2.5) — Architecture Authority
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Version, Status, Scope, Last updated | Metadata |
| **v2.4 Spec Summary** | S1–S10 bullet list | Feature status registry |
| **Table of Contents** | Sections 1–8 | Navigation |
| **Documentation Map** | 7-row table | Cross-doc navigation |
| **§1 Assumptions** | A1–A12 | System-scope constraints |
| — *A1–A4* | Single competition, tabular only, Zindi platform, supervised only | Scope boundaries |
| — *A5* | No hardcoded competition strings | Zero-literal rule |
| — *A6* | `SKILL_STATE` as execution state SOT + state/reports boundary rule + examples | State management contract |
| — *A7* | Universal OOF contract | CV/OOF contract anchor |
| — *A8* | Spatial = group signals | Spatial routing rule |
| — *A9* | Sidecar is non-blocking + consumption code pattern | Sidecar contract |
| — *A10* | Pinned environment, pip-compile workflow | Env reproducibility |
| — *A11* | Multi-target is config-declared, never inferred | Multi-target detection rule |
| — *A12* | Pseudo-label recombination policy mandatory for mixed-task | Multi-target augmentation rule |
| **§2 Core Architectural Principles** | Principles 1–6 | Design rules |
| — *P1* | Three-lens philosophy | Decision framework |
| — *P2* | Config boundary + temporal lock table + Phase 1 mutable window | Config write governance |
| — *P3* | OOF contract flow diagram + OOF Record Schema (JSON) + secondary_metrics + regression lifecycle + composite score + composite variance threshold | OOF & schema specifications |
| — *P4* | Dependency chain enforcement | Phase sequencing rule |
| — *P5* | Feedback loops over blind iteration | Variant generation rule |
| — *P6* | Human gates — 5 keys + CV strategy override mechanism ([C] and [D] paths + state schema) | Gate contract + override protocol |
| **§3 Preflight Validation** | INIT mode + ENFORCE mode | Session startup validation |
| — *INIT Mode* | Permitted skills, checks performed, output path | Bootstrap-phase behavior |
| — *ENFORCE Mode* | Config completeness, state integrity, OOF contract, architecture integrity, Zindi compliance, human gates checks | Post-Phase-1 validation |
| **§4 Phase Architecture** | Phases 1 → 2A → 2B → 3A → 3B → 4 | Per-phase specs |
| — *Phase 1* | `challenge_config.json` required layout (full JSON) + skill_03 breakout + skill_04 state schema + skill_05 CV decision tree + three-lens check + gate checklist | Fingerprint + config lock |
| — *Phase 2A* | `policy_gate()` + skill_06 imputation pipeline + three-lens check + gate checklist | Data cleaning |
| — *Phase 2B* | skill_08 anchor contract + multi-target extension + skill_07 engineering rules engine + two-mode contract details + three-lens check + gate checklist | Baseline + feature search |
| — *Phase 3A* | skill_10 SHAP contract + active gate logic + S6 MI audit + skill_09 calibration + skill_12 metric outputs + NB corrected variance + three-lens check + gate checklist | Generalisation audit |
| — *Phase 3B* | skill_11 gate (5 conditions, full threshold branching) + skill_21 pseudo-label (guard conditions + on-pass full contract + rollback path + on-fail schema) + skill_13 oracle fusion + three-lens check + gate checklist | Promotion + fusion |
| — *Phase 4* | skill_14 inference validation schema + skill_16 budget protocol + skill_17 governance outputs + skill_22 sign-off checklist + three-lens check + gate checklist | Governance |
| **Plugin Contract** | FeatureExtractor ABC + config rules | Multi-target plugin interface |
| **§5 Research Sidecar** | Sidecar state interface + recommendation schema + consumption rules + trigger schedule table + skill_00 specific triggers + **Cross-Competition History Log Schema** (`history_log.jsonl`) | Sidecar architecture |
| **§6 Reproducibility Contract** | R1–R5 + R6 | Reproducibility requirements |
| — *R1* | Three-seed pattern | Seed discipline |
| — *R2* | Bit-identical rerun | Determinism |
| — *R3* | No custom packages | Package discipline |
| — *R4* | Submission reproducible from config+state | Governance reproducibility |
| — *R5* | Carbon tracking — telemetry schema, carbon formula, mandatory/exempt skills | Compute impact |
| — *R6* | Computed artifact fingerprinting — 3-tier tolerance bands, status | Artifact integrity |
| **§7 Known Gaps Registry** | Resolved table (C1, GAP-4, S1–S5, C2/C4/M6/DRIFT-3 confirmed resolved) + Open gaps (S6, S7-skill09, S8, S10, skill_18/20 root writes, preflight MT-OOF, GAP-3, R5-telemetry-aggregate) | Code/spec delta tracker |

---

### AGENTS.md — Agent Implementation Guide
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | For use with, paired document, last updated, verification status note | Metadata |
| **Role and Scope** | Agent identity, resolution order (code > SoT > AGENTS.md) | Agent role definition |
| **Verification Status** | CONFIRMED / TARGET / UNVERIFIED tag framework + grep check pattern | Claim reliability system |
| **Repository Ground Truth** | 9-row fact table + "On the skill module count claim" paragraph | Confirmed file/function locations |
| **The Source of Truth Is Authoritative** | State contracts, OOF contract, anchor baseline key, config temporal lock ([RESOLVED — v2.5] C1), no hardcoded strings (DRIFT-1 resolved), no AutoML, oracle_fusion_core shim confirmed | Hard rule reminders + resolved risk notes |
| **Safe State Access Patterns** | 7 code blocks: CV override, pseudo-label retraining check, anchor challenge check, three-way baseline precedence, drift threshold, sidecar recommendations, EDA target_std + [RESOLVED — v2.5] M6 note | Mandatory .get() patterns |
| **Threshold and Metric Conventions** | Fold score variance (ddof=1 code), effective gate margin + variance threshold (branching logic table), metric direction (code), correlation in skill_13 (code + multi-target gap note) | Threshold computation rules + code |
| **OOF Output Schema** | 5-bullet summary + SOT reference link | Schema quick-reference |
| **Augmented OOF Namespace Contract** | Hard-error code block + rollback rule + SOT link | Pseudo-label write discipline |
| **SHAP Computation Rules** | Per-fold SHAP loop code + single-feature fallback code + SOT link | SHAP implementation pattern |
| **Two-Mode Feature Contract** | cv/inference mode code + structural features note + SOT link | Feature computation modes |
| **Seed Discipline** | 3-seed code block | Seed implementation |
| **Human Gate Keys** | 5-key schema + Gate 2 per-branch check code + [RESOLVED — v2.5] C4 + legacy key warning | Gate key patterns + resolved risks |
| **SKILL_STATE.json vs reports/** | Design rule + "ask before adding" test + confirmed boundary table (skill_03/04/10/15) + categorized subdirectory list + reader/writer path rule | State/report routing |
| **Budget Guard in skill_16** | 3-tier corrected code (HardAbortException paths, budget_warning writes) + datetime.utcnow deprecation warning | Corrected budget guard behavior |
| **preflight_enforce.py — What It Should Check** | Required check list + multi-target OOF tag risk + regex check command | Preflight extension guide |
| **Skill File Conventions** | Naming convention, run() signature, stateless rule, no internal CV, no config writes | Skill authoring standards |
| **What to Do When Unsure** | 8 stop-and-ask triggers | Escalation rules |
| **Environment and Package Rules** | 5 package discipline rules + pip-compile diff warning | Dependency management |
| **v2.3, v2.4 & v2.5 Refactor — Completed Items** | v2.5 gap closures (S5, C4, M6, DRIFT-3, C2, preflight path, S7/skill_12) + v2.4 S1–S10 ✅ + v2.3 DRIFT/GAP ✅ + test coverage | Completed work log |
| **Open Known Gaps** | 11 numbered gaps (S6, S7-skill09, S8, S10, R5-telemetry-aggregate, skill_18/20 root writes, preflight MT-OOF, GAP-3, regression pseudo-label, two-mode static verify, drift threshold) | Outstanding work + freeze list |
| **Footer** | SoT pairing note, maintainer | Metadata |

---

### README.md — Project Entry Point
| Section | Subsections | Information Category |
|---|---|---|
| **Title + intro** | One-liner description + resolution order note (runtime > SoT > AGENTS) | Project identity |
| **System properties** | 6 bullet technical claims (phase-gated, CV factory, OOF contract, two-mode, zero literals, no AutoML) | Key design properties |
| **Architecture Overview** | Core Principles (6-item list) + 4 Main Phases (one-line diagram) + SOT link | High-level architecture summary |
| **Project Structure** | Directory tree (competitions/, docs/, zindian/, tabula/, scripts/, tests/) | Repo layout |
| **Quick Start** | 6 steps: read docs → install → init ledger → run tests → use CLI → run Phase 1 sim | Onboarding sequence |
| — *Dependencies* | venv activation (Unix/Windows), pip install, optional pip-compile workflow | Environment setup |
| **Documentation** | 6-row table linking all docs | Navigation hub |
| **Security & Compliance** | Data integrity (MD5, atomic writes, config lock, zero literals) + Zindi compliance (AutoML scan, seeds, probabilities, budget) | Compliance summary |
| **Testing** | pytest invocation, covered modules, specific test areas | Test execution guide |
| **Contributing** | 4-step workflow (Design → Implement → Test → Document) + skill template Python code | Contributor guide |
| **Support** | 4 links (overview, SoT, troubleshooting, license) | Help navigation |
| **License** | Apache 2.0 | Legal |
| **Version History** | v2.0–v2.5 table with highlights | Changelog summary |

---

### docs/orchestrator_overview.md — Non-Technical Overview
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Version 2.5, Status, Last Updated | Metadata |
| **Documentation Landscape** | 9-row table mapping all docs to their owned content (includes document_map.md) | Cross-doc navigation |
| **Non-Technical Overview** | — | Plain-language system explanation |
| — *What is Zindian Orchestrator?* | Problem description + 6 problems it solves | Value proposition |
| — *Core Philosophy: Three Lenses* | General / Specific / Generalization table | Decision framework (non-technical) |
| — *The Journey: 4 Main Phases* | Phase 1–4 plain-English summaries with duration, output, key activities | Phase walkthrough |
| — *Key Safety Features* | Human gates table (5 gates) + Reproducibility contract requirements + No AutoML policy | Safety explainer |
| — *Special Features* | Carbon tracking (metrics + goal) + Pseudo-labeling (6 guard conditions + goal) + Multi-target support (how it works) | Feature highlights |
| — *What Makes This Valuable?* | 4-bullet value list | Value summary |
| — *What It's NOT* | 4-bullet anti-pattern list | Scope boundaries |
| — *Conceptual Analogy: Is This RL?* | RL structural parallels table + where it differs + two feedback mechanisms (cross-competition replay, Bayesian thresholds) + link to SOT §5 | Conceptual framing |
| — *Success Metrics* | Competition performance + operational efficiency + risk management | KPIs |
| **Technical Reference** | 8-row table linking all docs with content descriptions (includes document_map.md) | Technical navigation |
| **Footer** | Document version 1.3, Orchestrator version 2.5, scope note | Metadata |

---

### docs/quick_start.md — Operational Walkthrough
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Title, scope note referencing SoT v2.5 | Metadata |
| **§1 Environment & Setup** | venv activation (Unix/Windows), pip install, test suite verification | Environment bootstrap |
| **§2 Bootstrapping & Multi-Tenancy** | Competition path resolution order (5 priority levels) + F4 ambiguity hard-fail rule + Zindi API slug requirement + bootstrap command + created directory structure tree + .env auto-set | Competition workspace creation |
| **§3 Competition Data Intake & Configuration** | Ingest data files (cp commands) + full challenge_config.json example JSON (spatial regression competition) | Data setup + config population |
| **§4 Initialize Competition Ledger** | init-ledger command + ledger path note | DuckDB setup |
| **§5 Preflight Compliance Engine** | preflight command + INIT mode description + ENFORCE mode details (schema completeness, human gate memory, AST audits, Section 1 assumptions audit) | Preflight walkthrough |
| **§6 Pipeline Phase Execution & Variant Management** | Per-phase CLI commands with inline comments + --variant behavior across Phase 2B/3A/3B/4 + full 25-skill matrix table (slot, phase, type, role) | Phase execution guide + skill inventory |
| **§7 Querying the Competition Ledger** | 5 ledger query commands | DuckDB query guide |
| **§8 CLI Command Quick Reference** | 21-command table (category, command, description) | CLI reference summary |
| **Footer** | SoT version v2.5, Last Updated | Metadata |

---

### docs/cli_integration_guide.md — CLI Developer Reference
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Title, purpose | Metadata |
| **Quick Start** | 3 setup options (explicit flag, env var, CWD) + module syntax fallback | CLI setup |
| **Competition Context Resolution** | 6-level priority order (explicit → CWD → env var → .env → auto-detect → error) + env override injection behavior | Context resolution contract |
| **Unified Console Commands (21)** | 4 groups (A/B/C/D) with full per-command syntax + description | Command reference |
| — *Group A: Intake & Init* | bootstrap, init-ledger, preflight, preflight-sim, sync | Workspace + compliance commands |
| — *Group B: Phase Execution* | phase (with --variant detail per phase), status, monitor, report | Pipeline execution commands |
| — *Group C: Reproducibility & Validation* | verify-state, verify-phase-b, write-oof-meta, compile-requirements, audit, audit-framework, check-deployment | Validation + audit commands |
| — *Group D: Submissions & Leaderboards* | submit, submissions, leaderboard, archive, ledger | Zindi interaction commands |
| **Externalized Skill Logging** | Storage location, Tee behavior, subprocess safety note | Log infrastructure behavior |
| **Command Development Pattern** | Minimal subparser template (Python) + checklist | CLI extension guide |
| **Testing Commands** | pytest commands for CLI/phase/submission tests + integration test pattern (subprocess.run pattern) | CLI test guidance |
| **Common CLI Patterns** | Read-only query pattern (Python) + state mutation pattern (Python) | Code patterns for CLI consumers |

---

### docs/ledger_architecture.md — DuckDB Experiment Tracking
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Version 2.5, Last Updated, Authority note | Metadata |
| **§1 Purpose** | One-paragraph scope statement | Scope |
| **§2 Schema Definition** | experiments table DDL (SQL) + submissions table DDL (SQL) with FK relationship | Database schema |
| **§3 Context Manager Lifecycle** | Correct usage pattern (Python with-block) + forbidden pattern + rationale (DuckDB file-level locking) | Connection management contract |
| **§4 Persistence Guarantees** | CHECKPOINT requirement + verification test (Python) | Durability contract |
| **§5 Multi-Process Safety** | Read-only safety note + in-process write lock (threading.Lock) + cross-process atomic directory lock pattern (Python) | Concurrency contract |
| **§6 Error Handling** | Database locked retry pattern (Python) + corruption recovery (bash) | Error recovery |
| **§7 Migration Protocol** | 4-step schema version migration process | Schema change guide |
| **§8 Performance Benchmarks** | 3-operation latency table (p50/p99) + measurement conditions | Performance reference |
| **§9 References** | Session log link, DuckDB docs URL, implementation file | External references |

---

### docs/troubleshooting_guide.md — Runtime Error Fixes
| Section | Problem Type | Information Category |
|---|---|---|
| **§1 Package Shadowing** | Root-level stubs shadowing site-packages (DuckDB, LightGBM, Google) | Symptom + cause + resolution |
| **§2 Ledger DB Durability & Locks** | Wrong path resolution, database is locked, mypy cache lock | Symptom + cause + resolution (path fix, context manager pattern, cache clear commands) |
| **§3 Submission & Budget Tracking** | Budget counter vs. actual upload discrepancy, stale OOF comments | Symptom + resolution (state write ordering, branch-specific key reading) |
| **§4 Competition Context Resolution Failures** | "No competition found" CLI errors | Resolution (4-priority lookup + all platform env var commands) |
| **§5 Network Isolation & Zindi API Failures** | ConnectionError, ImportError: No module named 'zindi' | Symptom + resolution (ZINDIAN_DISABLE_NETWORK, credential check) |
| **§6 CI Pipeline & Dependency Issues** | pip-compile TypeError, Pyright reportMissingImports in CI | Problem + cause + resolution (pinned versions, venv pre-population) |
| **§7 SKILL_STATE Sidecar Externalization** | scores/ directory behavior, pointer structure, hydration/fallback | Architecture explanation (not a bug — expected behavior documentation) |

---

### docs/reporting_logging_audit.md — Codebase Audit Report
| Section | Subsections | Information Category |
|---|---|---|
| **Header** | Status, Last verified, Scope, Method | Metadata |
| **Executive Summary** | 4 corrected claims + decision record (Rec B selected, Rec A partially applied) | Findings summary |
| **Part 1 — Report Folder Flooding** | Dual-write table (4 skills with line numbers) + measured flood stats (59 files, 33 preflight, duplicated diagnostics, pseudo-label CSVs at root) | Code-verified investigation |
| **Part 2 — logs/ Organization** | Confirmed flat per-skill layout (L246–253 citations) + assessment | Log structure investigation |
| **Part 3 — SWOT by Phase** | Phase 1/2/3/4 SWOT tables | Phase-level quality analysis |
| — *Phase 1 SWOT* | Skill 05 placeholder bug (live defect, L366–375) + spatial/temporal auto-detect gap | Active defect + gap |
| — *Phase 2 SWOT* | Variant feature summary gap + Two-Mode contract threat | Gap + threat |
| — *Phase 3 SWOT* | Fold-only SHAP strength + 1-SE gate strength + std==0 warn-only gap + calibration opportunity + pseudo-label cap mitigant | Balanced SWOT |
| — *Phase 4 SWOT* | Governance gate-block strength + missing per-target metrics + no git tagging + rate limit threat | Balanced SWOT |
| **Part 4 — Recommendations** | Rec B (session-based logs — selected/implemented) + Rec A (root flooding — writer migration done, 3 remaining items) | Action decisions + residual work |
| **Part 5 — Action Plan** | Track 1 (log restructure — 3 steps) + Track 2 (report cleanup — 4 steps) + test count correction | Concrete next steps |
| **Part 6 — Corrected Claim Log** | 9-row table (original claim, verdict, correction) | Fact correction record |
| **Footer** | Verification method, SoT v2.5, maintainer | Metadata |

---

## 2. Cross-Document Information Map

The following map defines the distinct focus of each file vs. shared content, preventing duplication.

| Information Type | SOT | AGENTS.md | Overlap? |
|---|---|---|---|
| **Architecture contracts** (A1–A12, P1–P6) | ✅ Canonical | References | None — AGENTS links to SOT |
| **Phase specs + skill contracts** | ✅ Canonical | Links | None post-v2.5 |
| **Gate checklists** | ✅ Canonical (in §4) | ❌ Removed | None |
| **OOF record JSON schema** | ✅ Canonical | 5-bullet summary | Minimal — summary only in AGENTS |
| **Safe `.get()` access code patterns** | ❌ Not here | ✅ Canonical | None — complementary |
| **Threshold branching logic** | ✅ In skill_11 spec | ✅ Standalone code | Small overlap — SOT embeds it in skill_11; AGENTS surfaces it as a standalone reference |
| **Known live risks** (S6, S7, S8, S10) | §7 canonical | ✅ Inline per section with impl constraints | Minimal — SOT is registry, AGENTS is impl guide |
| **Completed work log** | §7 as gap registry | v2.3/v2.4 ✅ lists | Small overlap — same facts, different framing |
| **Repository file/function locations** | ❌ | ✅ Ground Truth table | None |
| **Per-skill completion checklists** | ❌ Removed in v2.5 | ❌ | None — removed from SOT; canonical source is code and AGENTS.md |
| **Agent role + escalation rules** | ❌ | ✅ Canonical | None |
| **Verification tag system** | ❌ | ✅ Canonical | None |

### Intentional Overlaps (Quick-Reference Copy)
1. **Threshold branching logic:** Appears fully in SOT §4 Phase 3B (skill_11) and is duplicated in AGENTS.md for quick lookup by coding agents, marked with a SOT cross-reference.
2. **v2.3/v2.4/v2.5 completed items:** SOT §7 resolved table is the canonical record; AGENTS.md `v2.3, v2.4 & v2.5 Refactor` section is the implementation-perspective done-list.

---

## 3. Full Ownership Matrix (All 9 Files)

| Information Type | README | overview | SOT | AGENTS | quick_start | cli_guide | ledger | troubleshoot | audit |
|---|---|---|---|---|---|---|---|---|---|
| Project intro + resolution order | ✅ | — | — | — | — | — | — | — | — |
| Key system properties (6 bullets) | ✅ | — | — | — | — | — | — | — | — |
| Non-technical phase summaries | — | ✅ | — | — | — | — | — | — | — |
| Three-lenses explainer (plain) | — | ✅ | — | — | — | — | — | — | — |
| Safety features plain summary | — | ✅ | — | — | — | — | — | — | — |
| Architecture contracts & schemas | — | — | ✅ | — | — | — | — | — | — |
| Phase specs & skill contracts | — | — | ✅ | — | — | — | — | — | — |
| OOF record schema (canonical) | — | — | ✅ | Link | — | — | — | — | — |
| Safe `.get()` access patterns | — | — | — | ✅ | — | — | — | — | — |
| Known live risks / open gaps (S6, S7, S8, S10) | — | — | — | ✅ | — | — | — | — | — |
| Agent role + escalation rules | — | — | — | ✅ | — | — | — | — | — |
| Repo file/function locations | — | — | — | ✅ | — | — | — | — | — |
| Competition bootstrap walkthrough | — | — | — | — | ✅ | — | — | — | — |
| Full 25-skill matrix table | — | — | — | — | ✅ | — | — | — | — |
| `config.json` example (spatial) | — | — | — | — | ✅ | — | — | — | — |
| Preflight operational modes | — | — | SOT §3 | — | ✅ (details)| — | — | — | — |
| 21 CLI commands (canonical) | — | — | — | — | Summary | ✅ | — | — | — |
| CLI extension & test patterns | — | — | — | — | — | ✅ | — | — | — |
| Context resolution (canonical) | — | — | — | — | Quick summary | ✅ | — | — | — |
| DuckDB schema & lifecycle | — | — | — | — | — | — | ✅ | — | — |
| Runtime error fixes | — | — | — | — | — | — | — | ✅ | — |
| SKILL_STATE sidecar externalization | — | — | — | — | — | — | — | ✅ | — |
| Report folder flooding analysis | — | — | — | — | — | — | — | — | ✅ |
| Phase SWOT & live defects | — | — | — | — | — | — | — | — | ✅ |
| RL analogy + feedback loop framing | — | ✅ | — | — | — | — | — | — | — |
| Open gap registry (canonical) | — | — | ✅ §7 | — | — | — | — | — | — |
| Open gap impl constraints | — | — | — | ✅ | — | — | — | — | — |
| Log restructure action plan | — | — | — | — | — | — | — | — | ✅ |
| Cross-doc navigation (primary) | Table | ✅ | Map | — | — | — | — | — | — |
| Version history / changelog | ✅ | — | Header | — | — | — | — | — | — |
| Contributing / skill template | ✅ | — | — | — | — | — | — | — | — |

### Key Complementary (Non-Redundant) Relationships
- **Context Resolution:** `quick_start.md` §2 lists the 5-level lookup table; `cli_integration_guide.md` details environment variable injection behaviors.
- **Preflight ENFORCE Checks:** SOT §3 holds the formal architecture specification, while `quick_start.md` §5 provides the CLI operational guides.
- **CLI Commands:** `quick_start.md` §8 presents a quick summary cheat sheet; `cli_integration_guide.md` defines command syntax and argument options.
