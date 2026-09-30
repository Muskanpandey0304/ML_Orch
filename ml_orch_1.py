#!/usr/bin/env python3
"""
================================================================================
CYBER MoE ORCHESTRATOR (DUAL EMBEDDING ENSEMBLE) - PLAN -> ROUTE -> GUIDE -> ANSWER
================================================================================
Target Server : http://10.8.1.104:8000/v1 (OpenAI Compatible)
Target Model  : bg-ft-experimental

Features & Optimizations:
  1. Fast Dual-Embedding Ensemble Routing:
     - Model A: BAAI/bge-large-en-v1.5
     - Model B: sentence-transformers/all-mpnet-base-v2
  2. Zero-latency parallel classification across 14 expert domains via pre-indexed L2 unit dot products
  3. Dynamic Agreement-Weighted Score Combination:
     - agreement = 1 - abs(score_A - score_B)
     - agreement > 0.8 => 0.60 * score_A + 0.40 * score_B
     - agreement > 0.5 => 0.55 * score_A + 0.45 * score_B
     - else            => 0.50 * score_A + 0.50 * score_B
  4. Real-time Parallel Execution (asyncio.create_task for Plan generation)
  5. Automatic Stop Token Enforcements (<|im_end|>, <|im_start|>, <|endoftext|>)
  6. Robust Reasoning & Think Tag Separation Regex
================================================================================
"""

import asyncio
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from openai import AsyncOpenAI

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
except ImportError:
    print("Missing required packages. Install via:")
    print("    pip install sentence-transformers numpy openai httpx")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CyberMoEOrchestrator")

SERVER_URL = os.environ.get("SERVER_URL", "http://10.8.1.104:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "bg-ft-experimental")

# ──────────────────────────────────────────────────────────────────────────────
# Embedding Model Configurations
# ──────────────────────────────────────────────────────────────────────────────
PRIMARY_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
SECONDARY_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Calibrated confidence thresholds for embeddings
EXPERT_TH = 0.18
VARIANT_TH = 0.25
VARIANT_MARGIN = 0.04

# Cross-Encoder Re-Ranker Configuration
CROSS_ENCODER_MODEL = "cisco-ai/SecureBERT2.0-cross_encoder"
RERANK_MARGIN_THRESHOLD = 0.08  # Trigger cross-encoder when bi-encoder margin < this
RERANK_TOP_K = 5                # Number of top candidates to re-rank
RERANK_CE_WEIGHT = 0.65         # Cross-encoder weight in score fusion
RERANK_BE_WEIGHT = 0.35         # Bi-encoder weight in score fusion


# ==============================================================================
# 1. OPTIMIZED & DECOUPLED EXPERT TAXONOMY
# ==============================================================================
EXPERTS: Dict[str, Dict[str, Any]] = {
    "E01": {
        "domain": "Code, Vulnerability & Systems Programming",
        "scope": "Find, explain, triage, and patch vulnerabilities in source code, algorithms, data structures (tries, trees, lists, arenas), concurrency primitives (spinlocks, mutexes, atomics, RCU), heap allocators (ptmalloc, tcache), cryptographic implementations (RSA, AES, ECC, HMAC, hashes, HKDF, PBKDF2, bcrypt, Argon2id, password hashing, key derivation, JWT algorithm flaws like 'none' and HMAC key confusion, ECB mode pattern leakage), and systems programming in C, Rust, Go, and Python. Also covers binary exploitation techniques (ROP, SROP, JOP, ret2libc, ret2dlresolve, stack pivots, heap spray, House of Force/Spirit/Lore/Einherjar, glibc pointer mangling, exit_function structures), kernel exploitation primitives (userfaultfd races, eBPF verifier bugs, dirty CoW, /dev/mem kernel patching, kASLR/FG-KASLR bypasses), CPU/OS security mitigations (NX/DEP, ASLR, CFG, CFI, PAC, BTI, KPTI, Shadow Stacks), and cryptographic protocol attacks (padding oracles, CBC bit-flipping, cookie role manipulation via bit-flip, nonce reuse, KRACK, DROWN, POODLE, SLOTH, Logjam, Lucky Thirteen, CRIME/BREACH, Bleichenbacher, length extension, Raccoon, invalid curve, fault injection on RSA-CRT, ECDSA nonce reuse, Dual EC DRBG, LCG cracking, small-subgroup confinement, constant-time modular exponentiation for side-channel resistance).",
        "inputs": "Source files, algorithm logic, concurrency specifications, heap traces, crash dumps, sanitizer traces (ASan/MSan), cryptographic primitives, SAST/DAST alerts, exploit technique descriptions, cryptographic protocol specifications.",
        "outputs": "Compilable source code, unified diff patches, root-cause explanations, cryptographic proofs, CWE mappings.",
        "boundary": "Compiled malware unpacking, software license crack/NOP patching, hardware/microarchitecture security (Spectre, Rowhammer, DRAM bit flips, JTAG, CAN bus, baseband, UEFI) and YARA rule generation go to E05; network packet captures and socket filters go to E02.",
    },
    "E02": {
        "domain": "Network, Packet Crafting & Traffic Telemetry",
        "scope": "Read and author network telemetry, raw socket packet crafting in C, eBPF/XDP/TC kernel filters, Suricata/Snort signatures, NetFlow/IPFIX analysis, and protocol decoding.",
        "inputs": "PCAP summaries, Zeek logs, NetFlow/IPFIX, raw packet bytes, eBPF XDP programs, Suricata/Snort rules.",
        "outputs": "Event timelines, Suricata/Snort rules, anomaly verdicts with evidence, eBPF C filters, protocol field decodes.",
        "boundary": "Host artifacts and memory dumps go to E06; SIEM/EDR detection rules go to E07. Cryptographic protocol ATTACKS and KDF theory (KRACK, DROWN, POODLE, SLOTH, Logjam, Dual EC DRBG, HKDF/Argon2id, TLS/SSL exploit math) go to E01. Automotive CAN bus logs, baseband modem IPC, and stripped binary reverse engineering go to E05.",
    },
    "E03": {
        "domain": "Infrastructure, Container & Cloud Security",
        "scope": "Audit, configure, and harden cloud infrastructure (AWS/Azure/GCP), container runtimes, Kubernetes manifests, NetworkPolicies, Dockerfiles, Terraform HCL, ARM/CloudFormation, OPA Rego, and Kyverno policies.",
        "inputs": "Terraform, CloudFormation, K8s YAML, Dockerfiles, OPA Rego, Kyverno policies, security group configurations.",
        "outputs": "Hardened IaC manifests, Rego/Kyverno policies, CIS benchmark remediations, security group rules.",
        "boundary": "Identity governance, RBAC entitlement reviews, and Active Directory / Entra ID graphs go to E08; application source code flaws go to E01.",
    },
    "E04": {
        "domain": "Threat Intelligence & TTP Mapping",
        "scope": "Turn unstructured intelligence, vulnerability exposure queues, and adversary briefs into structured, prioritized, ATT&CK-mapped output.",
        "inputs": "Threat reports, email headers, alert queues, sandbox summaries, CVE advisories, OSINT bulletins.",
        "outputs": "STIX 2.1 JSON bundles, ATT&CK mappings with justification, triage verdicts, executive threat briefs.",
        "boundary": "Rule authoring goes to E07/E02; binary reverse engineering goes to E05.",
    },
    "E05": {
        "domain": "Malware Analysis, Reverse Engineering & Hardware Security",
        "scope": "Reason about compiled malicious binaries (PE/ELF), disassembly (x86/x64/ARM/MIPS), binary patching (NOP-ing jump instructions, software license check bypass), deobfuscation (PowerShell/VBScript/macros), anti-analysis tricks, and author YARA rules. Also covers firmware reverse engineering (UEFI/UEFITool, U-Boot, squashfs extraction, binwalk, SPI flash dumps, firmware signing verification), hardware security analysis (JTAG/SWD debugging, voltage/clock fault injection, differential power analysis/DPA, DMA attacks over PCIe/Thunderbolt, cold boot attacks, Rowhammer DRAM bit flips, microarchitectural execution leaks like Spectre/Meltdown, read-out protection bypass), IoT and embedded device RE (microcontroller flash dumping, I2C/EEPROM, automotive CAN bus packet injection, Zigbee/802.15.4, baseband processor IPC), CPU architecture security (ARM TrustZone, Intel SGX, Secure Boot chains, Apple SEP), and reverse engineering tooling (Ghidra scripting, IDAPython, Capstone, Unicorn engine, symbol table inspection of stripped MIPS/ELF binaries).",
        "inputs": "PE/ELF headers, section tables, disassembly/Ghidra output, obfuscated scripts, dynamic sandbox logs, firmware images, SPI/JTAG dumps, hardware captures, IoT protocol frames, automotive CAN bus packet logs, stripped ELF binaries, microcontroller memory dumps.",
        "outputs": "Capability summaries, ATT&CK behavior maps, family verdicts, YARA rules, deobfuscated scripts, firmware analysis reports, hardware vulnerability assessments.",
        "boundary": "Source-level code vulnerabilities, heap allocator internals, binary exploitation TECHNIQUES (ROP, SROP, ret2dlresolve, glibc pointer mangling, /dev/mem kernel patching, ARM BTI/PAC mitigation theory), and cryptographic protocol attacks / KDF math go to E01.",
    },
    "E06": {
        "domain": "Digital Forensics & Incident Response (DFIR)",
        "scope": "Scope confirmed security incidents, interpret host evidence (Event IDs, Registry, Prefetch, Amcache), analyze memory dumps (Volatility), reconstruct timelines, and drive containment playbooks.",
        "inputs": "Windows event logs (4624, 7045, 4688), disk/memory artifacts, Volatility output, incident context.",
        "outputs": "Forensic incident reports, unified event timelines, root-cause verdicts, containment action lists.",
        "boundary": "Proactive hypothesis-driven hunting goes to E09; SIEM/Sigma rule authoring goes to E07. Firmware extraction, hardware memory attacks (cold boot, DMA, Rowhammer), UEFI analysis, and microcontroller flash dumping go to E05.",
    },
    "E07": {
        "domain": "Detection Engineering & SIEM Analytics",
        "scope": "Author, tune, and validate host and SIEM detections (Sigma, SPL, KQL, EQL), normalize log sources, and analyze detection coverage gaps.",
        "inputs": "Behavior descriptions, log samples, existing rules with false positives, ATT&CK techniques, schema definitions.",
        "outputs": "Sigma YAML, Splunk SPL, Microsoft Sentinel KQL, Elastic EQL queries, OCSF field mappings, tuned rules.",
        "boundary": "Packet-level detection and eBPF go to E02; proactive threat hunting hypotheses go to E09.",
    },
    "E08": {
        "domain": "Identity, Access & Authentication Governance",
        "scope": "Analyse identity posture, IAM policies, RBAC roles, Active Directory / Entra ID graphs, token security (OAuth/JWT/SAML), and authentication anomalies (AiTM, impossible travel).",
        "inputs": "IAM policies, Entra ID graphs, entitlement exports, sign-in/auth logs, Golden SAML/Kerberos artifacts.",
        "outputs": "Over-privilege findings, attack paths with fixes, Conditional Access policies, token security verdicts.",
        "boundary": "Cloud resource configurations, Kubernetes network policies, and container runtimes go to E03. Code-level cryptographic JWT implementation flaws ('none' algorithm, HMAC secret misuse) and cookie cipher bit-flipping go to E01.",
    },
    "E09": {
        "domain": "Threat Hunting & Behavioral Baselines",
        "scope": "Generate and test hunt hypotheses, investigate statistical anomalies against entity baselines, and plan proactive hunts.",
        "inputs": "ATT&CK techniques, threat intelligence feeds, entity baseline histories, detection inventories.",
        "outputs": "Hypotheses with hunt queries, baseline deviation assessments, proactive hunt plans.",
        "boundary": "Confirmed forensic incidents hand off to E06; operational detection rules hand off to E07.",
    },
    "E10": {
        "domain": "AppSec & Software Supply Chain",
        "scope": "Security review inside the development pipeline (CI/CD workflows, SBOMs, SCA alerts, secrets scanning, STRIDE threat modeling, and fuzzing harnesses).",
        "inputs": "PR diffs, GitHub Actions / GitLab CI YAML, SBOMs, advisory data, SCA reports, secret scanner hits.",
        "outputs": "Inline review comments, dependency reachability verdicts, STRIDE models, fuzzing harness sketches.",
        "boundary": "Deep root-cause memory/code flaws go to E01.",
    },
    "E11": {
        "domain": "GRC & Security Risk Scoring",
        "scope": "Map security controls to compliance frameworks (ISO 27001, NIST CSF, PCI-DSS, SOC 2), score risks, and answer regulatory questions.",
        "inputs": "Control requirements, audit evidence, policy drafts, asset exposure context.",
        "outputs": "Control mappings, compliance policy drafts, risk register entries, grounded regulatory answers.",
        "boundary": "Technical code and cloud infrastructure implementation goes to E03/E01.",
    },
    "E12": {
        "domain": "Orchestrator, Router & Synthesis",
        "scope": "Route tasks across domains, plan multi-step workflows, manage context budgets, and fuse multi-expert findings.",
        "inputs": "User objective, tool schemas, truncated history, expert outputs, environment state.",
        "outputs": "Expert routing decisions, ordered execution plans, state summaries, synthesized final answers.",
        "boundary": "Performs no direct domain analysis itself.",
    },
    "E13": {
        "domain": "Data Security & Privacy",
        "scope": "Classify sensitive data (PII, PCI, HIPAA), author DLP regex rules, and analyze data exfiltration risks.",
        "inputs": "Database schemas, sample data records, DLP alerts, network egress records.",
        "outputs": "Data sensitivity labels, DLP rule regexes, exfiltration verdicts, data retention policies.",
        "boundary": "Network-level socket packet inspection goes to E02.",
    },
    "E14": {
        "domain": "OT, ICS & SCADA Security",
        "scope": "Monitor, analyze, and harden operational technology where safety and physical uptime outrank confidentiality.",
        "inputs": "ICS protocol captures (Modbus TCP, DNP3, S7, OPC-UA), historian telemetry, PLC logic, Purdue model diagrams.",
        "outputs": "Unsafe command findings, OT anomaly detections, network segmentation fixes, ATT&CK-for-ICS mappings.",
        "boundary": "IT enterprise-side forensic response goes to E06.",
    },
}

# ==============================================================================
# 2. VARIANT TAXONOMY
# ==============================================================================
VARIANTS: Dict[str, Dict[str, Any]] = {
    "V1-1": {"expert": "E01", "task": "Vulnerability detection", "input": "Function or file, optionally with scanner alert"},
    "V1-2": {"expert": "E01", "task": "Root-cause analysis", "input": "Crash or sanitizer trace plus source"},
    "V1-3": {"expert": "E01", "task": "Reachability assessment", "input": "Vulnerable function plus call context"},
    "V1-4": {"expert": "E01", "task": "Patch generation", "input": "Vulnerable function plus CWE"},
    "V1-5": {"expert": "E01", "task": "Patch review", "input": "Original code, proposed patch"},
    "V1-6": {"expert": "E01", "task": "SAST alert triage", "input": "Scanner finding plus surrounding code"},
    "V1-7": {"expert": "E01", "task": "CVE to CWE mapping", "input": "CVE description text"},
    "V1-8": {"expert": "E01", "task": "Binary and decompiled source analysis", "input": "Decompiler pseudo-code or C/C++ disassembly snippet analyzed for source vulnerabilities"},
    "V1-9": {"expert": "E01", "task": "Exploit technique analysis", "input": "ROP chain, SROP, heap exploitation (House of Force/Spirit/Lore/Einherjar), stack pivot, ret2libc, ret2dlresolve, glibc pointer mangling, /dev/mem kernel patching primitive, or software memory mitigations (NX/DEP, ASLR, CFG, CFI, PAC, BTI, KPTI, kASLR)"},
    "V1-10": {"expert": "E01", "task": "Cryptographic protocol attack analysis", "input": "TLS/SSL attack (KRACK, DROWN, POODLE, SLOTH, Logjam, Lucky Thirteen, CRIME, BREACH, Raccoon), WPA2 attack, cross-protocol cryptographic vulnerability, or Dual EC DRBG backdoor mathematical state recovery"},
    "V1-11": {"expert": "E01", "task": "Cryptographic primitive flaw analysis", "input": "ECDSA nonce reuse, RSA-CRT fault injection, padding oracle, CBC bit-flip (cookie role manipulation), Bleichenbacher, LCG/PRNG weakness, small-subgroup DH attack, invalid curve ECC, length extension, nonce reuse in AES-GCM, constant-time modular exponentiation, ECB image pattern leak, JWT 'none' algorithm / HMAC secret misuse, or password KDF comparison (HKDF, PBKDF2, bcrypt, Argon2id)"},
    "V2-1": {"expert": "E02", "task": "Flow and log analysis", "input": "Zeek conn/dns/http logs or NetFlow summary"},
    "V2-2": {"expert": "E02", "task": "PCAP summarisation", "input": "PCAP converted to session summary"},
    "V2-3": {"expert": "E02", "task": "Protocol decoding", "input": "Hex or byte stream"},
    "V2-4": {"expert": "E02", "task": "IDS signature authoring", "input": "Malicious requests or flows"},
    "V2-5": {"expert": "E02", "task": "Beaconing and C2 detection", "input": "Flow timing and volume features"},
    "V2-6": {"expert": "E02", "task": "DGA and DNS tunnelling", "input": "DNS query logs"},
    "V2-7": {"expert": "E02", "task": "Exfiltration detection", "input": "Egress flow and volume data"},
    "V2-8": {"expert": "E02", "task": "TLS and fingerprint reasoning", "input": "TLS handshake data, JA3/JA4 hashes"},
    "V3-1": {"expert": "E03", "task": "IaC misconfiguration audit", "input": "Terraform, CloudFormation or K8s YAML"},
    "V3-2": {"expert": "E03", "task": "Cloud posture assessment", "input": "Resource configuration export"},
    "V3-3": {"expert": "E03", "task": "Firewall and SG synthesis", "input": "Requirement statement"},
    "V3-4": {"expert": "E03", "task": "IAM privilege reduction", "input": "Policy plus observed usage"},
    "V3-5": {"expert": "E03", "task": "Container and K8s hardening", "input": "Manifest or Dockerfile"},
    "V3-6": {"expert": "E03", "task": "Policy-as-code generation", "input": "Control requirement in prose"},
    "V3-7": {"expert": "E03", "task": "CIS benchmark remediation", "input": "Config plus failing control ID"},
    "V4-1": {"expert": "E04", "task": "IOC extraction", "input": "Report or bulletin text"},
    "V4-2": {"expert": "E04", "task": "IOC enrichment and pivoting", "input": "Indicator plus context"},
    "V4-3": {"expert": "E04", "task": "ATT&CK technique mapping", "input": "Behaviour description or report excerpt"},
    "V4-4": {"expert": "E04", "task": "Actor profiling", "input": "Multiple reports or TTP sets"},
    "V4-5": {"expert": "E04", "task": "Campaign correlation", "input": "Several alerts or reports"},
    "V4-6": {"expert": "E04", "task": "Phishing triage", "input": "Headers plus SPF/DKIM/DMARC results"},
    "V4-7": {"expert": "E04", "task": "CVE prioritisation", "input": "CVE plus asset and exposure context"},
    "V4-8": {"expert": "E04", "task": "Report to brief", "input": "Long report"},
    "V5-1": {"expert": "E05", "task": "Static triage", "input": "PE/ELF headers, imports, strings"},
    "V5-2": {"expert": "E05", "task": "Sandbox behaviour summary", "input": "Dynamic analysis log"},
    "V5-3": {"expert": "E05", "task": "Family classification", "input": "Feature vector or behaviour profile"},
    "V5-4": {"expert": "E05", "task": "Deobfuscation reasoning", "input": "Obfuscated PowerShell, JS or macro"},
    "V5-5": {"expert": "E05", "task": "YARA rule generation", "input": "Sample features or analysis"},
    "V5-6": {"expert": "E05", "task": "Technique extraction", "input": "Behaviour log or analysis"},
    "V5-7": {"expert": "E05", "task": "Firmware and hardware reverse engineering", "input": "UEFI image, SPI flash dump, U-Boot partition, squashfs firmware blob, binwalk output, microcontroller flash with read-out protection, or stripped MIPS/ELF symbol table disassembly"},
    "V5-8": {"expert": "E05", "task": "Hardware security analysis", "input": "JTAG/SWD interface, voltage/clock fault injection, DPA side-channel, DMA attack over PCIe/Thunderbolt, cold boot RAM remanence, Rowhammer DRAM bit flips, Spectre/Meltdown speculative execution, ARM TrustZone, Intel SGX, Apple SEP (Secure Enclave Processor), Secure Boot chain, binary NOP patching for software license bypass, or IDAPython / Ghidra disassembly scripts"},
    "V5-9": {"expert": "E05", "task": "IoT and embedded protocol analysis", "input": "Zigbee/802.15.4 frames, automotive CAN bus packet log injection, baseband processor IPC, I2C/EEPROM dump, Unicorn/Capstone emulation of embedded code"},
    "V6-1": {"expert": "E06", "task": "Host artifact interpretation", "input": "Registry, prefetch, AmCache, EDR data"},
    "V6-2": {"expert": "E06", "task": "Memory forensics triage", "input": "Volatility plugin output"},
    "V6-3": {"expert": "E06", "task": "Timeline reconstruction", "input": "Multi-source events"},
    "V6-4": {"expert": "E06", "task": "Root cause and initial access", "input": "Evidence bundle"},
    "V6-5": {"expert": "E06", "task": "Scope and blast radius", "input": "Confirmed findings plus asset context"},
    "V6-6": {"expert": "E06", "task": "Containment recommendation", "input": "Scope plus business context"},
    "V6-7": {"expert": "E06", "task": "Incident report writing", "input": "Findings plus timeline"},
    "V7-1": {"expert": "E07", "task": "Sigma rule authoring", "input": "Behaviour description or log sample"},
    "V7-2": {"expert": "E07", "task": "SPL and KQL authoring", "input": "Behaviour plus target schema"},
    "V7-3": {"expert": "E07", "task": "Rule tuning", "input": "Rule plus false-positive examples"},
    "V7-4": {"expert": "E07", "task": "Correlation design", "input": "Multi-stage attack scenario"},
    "V7-5": {"expert": "E07", "task": "Log normalisation", "input": "Raw log line plus target schema"},
    "V7-6": {"expert": "E07", "task": "Coverage gap and tests", "input": "Detection inventory or a rule"},
    "V8-1": {"expert": "E08", "task": "Entitlement review", "input": "Entitlement or role export"},
    "V8-2": {"expert": "E08", "task": "Attack path analysis", "input": "AD or Entra graph excerpt"},
    "V8-3": {"expert": "E08", "task": "Authentication anomaly", "input": "Sign-in and auth logs"},
    "V8-4": {"expert": "E08", "task": "Access policy authoring", "input": "Requirements in prose"},
    "V8-5": {"expert": "E08", "task": "Privileged account review", "input": "Service and admin account usage"},
    "V9-1": {"expert": "E09", "task": "Hypothesis generation", "input": "ATT&CK technique or intel"},
    "V9-2": {"expert": "E09", "task": "Hunt query generation", "input": "Hypothesis plus data schema"},
    "V9-3": {"expert": "E09", "task": "Baseline reasoning", "input": "Entity history"},
    "V9-4": {"expert": "E09", "task": "Anomaly investigation", "input": "Baseline plus anomalous signal"},
    "V10-1": {"expert": "E10", "task": "Diff security review", "input": "Pull request diff"},
    "V10-2": {"expert": "E10", "task": "Dependency reachability", "input": "SBOM plus advisory"},
    "V10-3": {"expert": "E10", "task": "Secrets triage", "input": "Scanner hit plus context"},
    "V10-4": {"expert": "E10", "task": "Threat modelling", "input": "Design or data-flow description"},
    "V10-5": {"expert": "E10", "task": "Fuzzing guidance", "input": "Target function or interface"},
    "V11-1": {"expert": "E11", "task": "Control mapping", "input": "Requirement or finding"},
    "V11-2": {"expert": "E11", "task": "Policy drafting", "input": "Topic plus scope"},
    "V11-3": {"expert": "E11", "task": "Risk scoring", "input": "Asset, threat and exposure"},
    "V11-4": {"expert": "E11", "task": "Evidence assessment", "input": "Evidence artifacts plus control"},
    "V11-5": {"expert": "E11", "task": "Compliance Q&A", "input": "Framework question"},
    "V12-1": {"expert": "E12", "task": "Expert routing", "input": "Task description"},
    "V12-2": {"expert": "E12", "task": "Tool invocation", "input": "Goal plus tool schemas"},
    "V12-3": {"expert": "E12", "task": "Multi-step planning", "input": "Objective plus environment state"},
    "V12-4": {"expert": "E12", "task": "Error recovery", "input": "Failed step plus error"},
    "V12-5": {"expert": "E12", "task": "Context summarisation", "input": "Long history or telemetry"},
    "V12-6": {"expert": "E12", "task": "Result synthesis", "input": "Several expert outputs"},
    "V12-7": {"expert": "E12", "task": "Abstention", "input": "Question with insufficient evidence"},
    "V13-1": {"expert": "E13", "task": "Data classification", "input": "Schema or sample records"},
    "V13-2": {"expert": "E13", "task": "DLP rule authoring", "input": "Requirement plus data types"},
    "V13-3": {"expert": "E13", "task": "Exfiltration assessment", "input": "Egress logs plus data context"},
    "V13-4": {"expert": "E13", "task": "Retention review", "input": "Data flow or retention config"},
    "V14-1": {"expert": "E14", "task": "ICS protocol analysis", "input": "Modbus, DNP3 or S7 capture"},
    "V14-2": {"expert": "E14", "task": "OT anomaly detection", "input": "Historian or process telemetry"},
    "V14-3": {"expert": "E14", "task": "Segmentation review", "input": "Zone and conduit diagram plus rules"},
    "V14-4": {"expert": "E14", "task": "ATT&CK for ICS mapping", "input": "OT incident description"},
}

EXPERT_INVARIANTS: Dict[str, List[str]] = {
    "E01": [
        "Never invent a CWE, line number, or severity that isn't actually evidenced by the code/trace given.",
        "Distinguish reachable vulnerabilities from theoretical ones before assigning severity.",
        "If proposing a fix, check it doesn't silently break existing behaviour.",
    ],
    "E02": [
        "Distinguish confirmed malicious traffic from merely anomalous/unusual traffic.",
        "State the evidence (fields, timing, volume) behind any verdict, not just the conclusion.",
        "Don't assume attribution from traffic patterns alone.",
    ],
    "E03": [
        "Prefer least-privilege and deny-by-default reasoning over permissive defaults.",
        "Call out blast radius of a misconfiguration, not just its existence.",
        "Don't recommend a fix that breaks stated functional requirements without flagging trade-offs.",
    ],
    "E04": [
        "Separate confirmed indicators from speculative/inferred ones.",
        "Map to ATT&CK only when behaviour matches the technique definition.",
        "Flag low-confidence attribution explicitly rather than asserting it.",
    ],
    "E05": [
        "Separate static evidence from dynamic (sandbox) evidence in reasoning.",
        "Don't assign a malware family without comparable/matching features.",
        "Explain intent behind obfuscated code; do not merely restate decoded text.",
    ],
    "E06": [
        "Never assert root cause or scope without an evidence chain connecting steps.",
        "Keep timeline ordering strictly evidence-based, not assumed.",
        "Separate confirmed compromise from suspected/possible compromise.",
    ],
    "E07": [
        "Consider false-positive risk of detection logic before presenting it as final.",
        "Ground rule logic in actual behaviour/log fields given, not generic boilerplate.",
        "Note any assumptions about log source/schema availability.",
    ],
    "E08": [
        "Distinguish excessive privilege from privilege that is actually exploitable.",
        "Consider blast radius of compromised identity before recommending fixes.",
        "Don't flag standard delegated admin patterns as findings without justification.",
    ],
    "E09": [
        "State the hypothesis being tested and why it's plausible before jumping to a verdict.",
        "Distinguish 'no evidence found' from 'confirmed benign'.",
        "Ground hunts in available data sources, not hypothetical telemetry.",
    ],
    "E10": [
        "Focus review on changes in the diff, not pre-existing codebase issues.",
        "Distinguish theoretically vulnerable dependencies from actually reachable ones.",
        "Verify potential secrets are not test/dummy values when context permits.",
    ],
    "E11": [
        "Ground control mappings in explicit framework text/citations.",
        "Distinguish compliant, partially compliant, and non-compliant statuses clearly.",
        "State the likelihood/impact reasoning behind risk scores.",
    ],
    "E12": [
        "Explicitly hand off tasks to target domain experts instead of forcing local generation.",
        "Order multi-step plans by logical dependency.",
        "Prefer explicit abstention over speculation when evidence is insufficient.",
    ],
    "E13": [
        "Base sensitivity classification on actual payload data, not field names alone.",
        "Distinguish policy violations from confirmed data exposure.",
    ],
    "E14": [
        "Weigh safety/uptime impact equal to or above confidentiality in recommendations.",
        "Flag risks of disrupting live physical processes for all recommended actions.",
    ],
}

PLAN_SYSTEM_PROMPT = (
    "You are an expert technical problem-solver. You will be given a task or question. "
    "Do NOT answer it yet. Instead, produce a short internal plan: "
    "(1) what is actually being asked, "
    "(2) what information/evidence is given vs. missing, "
    "(3) what approach, method, or pseudocode steps would correctly solve it, "
    "(4) what mistakes or pitfalls to avoid. "
    "Be concise and concrete. Do not write the final answer itself."
)

FINAL_FORMAT_NOTE = (
    "\n\nThere is NO fixed output format required. Answer in whatever structure "
    "(prose, list, code, diff, table, JSON, etc.) genuinely best fits the request."
)


# ==============================================================================
# 3. DUAL EMBEDDING ENSEMBLE CLASSIFIER (Agreement-Weighted Model Combo)
# ==============================================================================
class DualEmbeddingEnsembleClassifier:
    def __init__(
        self,
        primary_model: str = PRIMARY_EMBEDDING_MODEL,
        secondary_model: str = SECONDARY_EMBEDDING_MODEL,
    ):
        logger.info(
            f"Loading Dual Embedding Ensemble: Model A='{primary_model}' | Model B='{secondary_model}'..."
        )

        self.embed_model_A = self._load_encoder(primary_model, FALLBACK_EMBEDDING_MODEL)
        self.embed_model_B = self._load_encoder(secondary_model, FALLBACK_EMBEDDING_MODEL)

        self.expert_ids = list(EXPERTS.keys())
        expert_corpus = [
            f"{eid}: {EXPERTS[eid]['domain']} | {EXPERTS[eid]['scope']} | Inputs: {EXPERTS[eid]['inputs']}"
            for eid in self.expert_ids
        ]

        # Pre-calculate unit normalized embeddings for zero-latency matrix operations
        vecs_A = self.embed_model_A.encode(expert_corpus, convert_to_numpy=True, normalize_embeddings=True)
        self.expert_embeddings_A = vecs_A.astype(np.float32)

        vecs_B = self.embed_model_B.encode(expert_corpus, convert_to_numpy=True, normalize_embeddings=True)
        self.expert_embeddings_B = vecs_B.astype(np.float32)

        self.variant_ids_by_expert: Dict[str, List[str]] = {}
        self.variant_embeddings_A: Dict[str, np.ndarray] = {}

        for eid in self.expert_ids:
            vids = [vid for vid, v in VARIANTS.items() if v["expert"] == eid]
            if not vids:
                continue
            corpus = [
                f"{vid}: {VARIANTS[vid]['task']} | Input: {VARIANTS[vid]['input']}"
                for vid in vids
            ]
            raw_v_A = self.embed_model_A.encode(corpus, convert_to_numpy=True, normalize_embeddings=True)
            self.variant_ids_by_expert[eid] = vids
            self.variant_embeddings_A[eid] = raw_v_A.astype(np.float32)

        logger.info(f"Dual Embedding Classifier ready with {len(self.expert_ids)} experts and {len(VARIANTS)} variants.")

        # Stage 2: Load SecureBERT 2.0 Cross-Encoder for re-ranking ambiguous cases
        logger.info(f"Loading Cross-Encoder re-ranker: '{CROSS_ENCODER_MODEL}'...")
        try:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            self.cross_encoder_available = True
            logger.info("Cross-Encoder re-ranker loaded successfully.")
        except Exception as exc:
            logger.warning(f"Cross-Encoder failed to load ({exc}). Re-ranking disabled, bi-encoder only.")
            self.cross_encoder = None
            self.cross_encoder_available = False

        # Pre-build rich expert descriptions for cross-encoder pairing
        self.expert_descriptions: Dict[str, str] = {}
        for eid in self.expert_ids:
            e = EXPERTS[eid]
            self.expert_descriptions[eid] = (
                f"{eid}: {e['domain']} — {e['scope']} | "
                f"Expected inputs: {e['inputs']} | "
                f"Out of scope: {e['boundary']}"
            )

    def _load_encoder(self, model_name: str, fallback_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(f"Failed to load model '{model_name}' ({exc}). Falling back to '{fallback_name}'...")
            return SentenceTransformer(fallback_name)

    def combine_embedding_scores(self, query: str) -> Dict[str, Any]:
        """
        Get embeddings from both Model A (bge-large) and Model B (mpnet-base).
        Calculate agreement between models: agreement = 1 - abs(score_A - score_B).
        Dynamically combine scores based on agreement.
        Return best expert and detailed scoring breakdown.
        """
        query_emb_A = self.embed_model_A.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
        query_emb_B = self.embed_model_B.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

        scores_A = np.dot(self.expert_embeddings_A, query_emb_A)
        scores_B = np.dot(self.expert_embeddings_B, query_emb_B)

        scores_all: Dict[str, Dict[str, float]] = {}

        for i, expert_id in enumerate(self.expert_ids):
            s_A = float(scores_A[i])
            s_B = float(scores_B[i])
            agreement = 1.0 - abs(s_A - s_B)

            if agreement > 0.8:
                combined = (0.6 * s_A) + (0.4 * s_B)
            elif agreement > 0.5:
                combined = (0.55 * s_A) + (0.45 * s_B)
            else:
                combined = (0.5 * s_A) + (0.5 * s_B)

            scores_all[expert_id] = {
                'score': combined,
                'agreement': agreement,
                'score_A': s_A,
                'score_B': s_B,
            }

        # Rank experts by combined score
        ranked_experts = sorted(scores_all.keys(), key=lambda eid: scores_all[eid]['score'], reverse=True)
        best_expert = ranked_experts[0]
        best_score = scores_all[best_expert]['score']

        second_expert = ranked_experts[1] if len(ranked_experts) > 1 else None
        second_score = scores_all[second_expert]['score'] if second_expert else 0.0
        margin = float(best_score - second_score)

        return {
            "best_expert": best_expert,
            "best_score": float(best_score),
            "margin": margin,
            "top2": [(best_expert, float(best_score)), (second_expert, float(second_score))],
            "scores_all": scores_all,
            "query_emb_A": query_emb_A,
        }

    def _cross_encoder_rerank(self, query: str, biencoder_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-rank top-K experts using SecureBERT 2.0 Cross-Encoder.
        Processes (query, expert_description) pairs through full self-attention
        to catch micro-nuances that bi-encoders miss between closely related domains.
        """
        scores_all = biencoder_result["scores_all"]

        # Get top-K experts by bi-encoder combined score
        ranked = sorted(scores_all.keys(), key=lambda eid: scores_all[eid]['score'], reverse=True)
        top_k = ranked[:RERANK_TOP_K]

        # Build (query, expert_description) pairs for cross-encoder
        pairs = [(query, self.expert_descriptions[eid]) for eid in top_k]

        # Cross-encoder prediction (raw logits)
        ce_raw_scores = self.cross_encoder.predict(pairs)

        # Sigmoid normalization to [0, 1]
        ce_scores = 1.0 / (1.0 + np.exp(-np.array(ce_raw_scores, dtype=np.float64)))

        # Fuse cross-encoder + bi-encoder scores
        fused_results: Dict[str, Dict[str, float]] = {}
        for i, eid in enumerate(top_k):
            be_score = scores_all[eid]['score']
            ce_score = float(ce_scores[i])
            fused = RERANK_CE_WEIGHT * ce_score + RERANK_BE_WEIGHT * be_score
            fused_results[eid] = {
                'fused_score': fused,
                'ce_score': ce_score,
                'be_score': be_score,
            }

        # Rank by fused score
        sorted_experts = sorted(
            fused_results.keys(),
            key=lambda eid: fused_results[eid]['fused_score'],
            reverse=True,
        )
        best_expert = sorted_experts[0]
        best_fused = fused_results[best_expert]['fused_score']

        second_expert = sorted_experts[1] if len(sorted_experts) > 1 else None
        second_fused = fused_results[second_expert]['fused_score'] if second_expert else 0.0

        logger.info(
            "[CrossEncoder Rerank] top-%d: %s → Winner: %s (fused=%.3f)",
            len(top_k),
            " | ".join(f"{eid}={fused_results[eid]['ce_score']:.3f}" for eid in top_k),
            best_expert,
            best_fused,
        )

        return {
            "best_expert": best_expert,
            "best_score": best_fused,
            "margin": float(best_fused - second_fused),
            "cross_encoder_score": fused_results[best_expert]['ce_score'],
            "top2": [(best_expert, best_fused), (second_expert, second_fused)],
            "fused_results": fused_results,
        }

    def route(self, query: str) -> Dict[str, Any]:
        res = self.combine_embedding_scores(query)
        expert_id = res["best_expert"]
        expert_conf = res["best_score"]
        margin = res["margin"]
        reranked = False
        ce_score = None
        routing_method = "dual_embedding_agreement"

        # Adaptive Gating: invoke cross-encoder only when bi-encoder margin is tight
        if self.cross_encoder_available and margin < RERANK_MARGIN_THRESHOLD:
            logger.info(
                "[Adaptive Gate] Margin=%.4f < %.2f → Triggering cross-encoder re-rank (bi-encoder pick was %s)",
                margin, RERANK_MARGIN_THRESHOLD, expert_id,
            )
            rerank_result = self._cross_encoder_rerank(query, res)
            expert_id = rerank_result["best_expert"]
            expert_conf = rerank_result["best_score"]
            margin = rerank_result["margin"]
            ce_score = rerank_result["cross_encoder_score"]
            reranked = True
            routing_method = "dual_embedding_+_cross_encoder_rerank"
        else:
            if not self.cross_encoder_available:
                logger.debug("[Adaptive Gate] Cross-encoder not available, using bi-encoder only.")
            else:
                logger.info(
                    "[Adaptive Gate] Margin=%.4f >= %.2f → Bi-encoder confident, skipping re-rank",
                    margin, RERANK_MARGIN_THRESHOLD,
                )

        # Variant task matching for chosen expert
        vids = self.variant_ids_by_expert.get(expert_id, [])
        best_v_id = None
        best_v_score = 0.0

        if vids:
            v_embeds = self.variant_embeddings_A[expert_id]
            v_scores = np.dot(v_embeds, res["query_emb_A"])
            v_order = np.argsort(v_scores)[::-1]
            best_v_id = vids[int(v_order[0])]
            best_v_score = float(v_scores[int(v_order[0])])

        tier = "STRICT" if (expert_conf >= EXPERT_TH and margin >= VARIANT_MARGIN) else "SOFT_GUIDED"

        return {
            "tier": tier,
            "expert_id": expert_id,
            "expert_conf": expert_conf,
            "margin": margin,
            "variant_id": best_v_id,
            "variant_conf": best_v_score,
            "top2": res.get("top2", []),
            "routing_method": routing_method,
            "reranked": reranked,
            "cross_encoder_score": ce_score,
        }


# ==============================================================================
# 4. OPTIMIZED ASYNC INFERENCE CLIENT (With Loop Prevention Penalties)
# ==============================================================================
class AsyncInferenceEngineClient:
    def __init__(self, server_url: Optional[str] = None, model_name: Optional[str] = None):
        self.server_url = server_url or os.environ.get("SERVER_URL", SERVER_URL)
        self.model_name = model_name or os.environ.get("MODEL_NAME", MODEL_NAME)
        self.http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            timeout=300.0,
        )
        self.client = AsyncOpenAI(
            base_url=self.server_url,
            api_key="not-needed",
            http_client=self.http_client,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        start = time.perf_counter()

        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            presence_penalty=0.25,
            frequency_penalty=0.35,
            max_tokens=max_tokens,
            stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"],
        )
        elapsed = time.perf_counter() - start
        raw_content = completion.choices[0].message.content or ""

        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        answer = raw_content

        if not reasoning:
            think_match = re.search(r"<think>(.*?)</think>", raw_content, re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                answer = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

        usage = getattr(completion, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        tps = (completion_tokens / elapsed) if elapsed > 0 and completion_tokens > 0 else 0.0

        return {
            "answer": answer,
            "reasoning": reasoning,
            "elapsed": elapsed,
            "tokens": completion_tokens,
            "tps": tps,
        }

    async def close(self):
        await self.http_client.aclose()


# ==============================================================================
# 5. CONCURRENT MASTER ORCHESTRATOR (DUAL EMBEDDING ENSEMBLE)
# ==============================================================================
class CyberMoEOrchestrator:
    def __init__(self, server_url: Optional[str] = None, model_name: Optional[str] = None):
        self.classifier = DualEmbeddingEnsembleClassifier()
        self.client = AsyncInferenceEngineClient(server_url=server_url, model_name=model_name)

    def _build_guided_prompt(self, route: Dict[str, Any], plan_text: str) -> str:
        tier = route["tier"]
        plan_block = f"Here is an initial approach/plan for this request:\n{plan_text}\n\nUse it as a starting point, but correct or extend it if it misses something.\n\n"

        if tier == "GENERIC" or not route.get("expert_id"):
            return (
                "You are a senior cybersecurity expert assistant. Answer clearly and accurately.\n\n"
                + plan_block
                + FINAL_FORMAT_NOTE
            )

        expert_id = route["expert_id"]
        e = EXPERTS[expert_id]
        invariants = "\n".join(f"- {i}" for i in EXPERT_INVARIANTS.get(expert_id, []))
        header = (
            f"You are a Senior Cybersecurity Specialist in the domain: {e['domain']}.\n"
            f"Scope: {e['scope']}\n"
            f"Out of scope (hand off instead of forcing an answer): {e['boundary']}\n\n"
            f"Things to keep in mind for this domain:\n{invariants}\n\n"
        )

        variant_id = route.get("variant_id")
        if not variant_id:
            return header + plan_block + FINAL_FORMAT_NOTE

        v = VARIANTS[variant_id]
        body = (
            f"This request most closely matches: {v['task']} ({variant_id}).\n"
            f"Typically the relevant input looks like: {v['input']}\n\n"
            + plan_block
            + "Never fabricate specifics (CVEs, line numbers, IOCs, hashes) not present in the "
            "user's input — say so explicitly if information is missing instead of guessing."
        )
        return header + body + FINAL_FORMAT_NOTE

    async def process_query(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Processing query: '{user_query[:60]}...'")

        # Step 1: Zero-Latency Dual Embedding Ensemble Classification
        route = await asyncio.to_thread(self.classifier.route, user_query)
        logger.info(
            f"Dual Embedding Route: expert={route['expert_id']} | conf={route['expert_conf']:.3f} | "
            f"margin={route['margin']:.3f} | method={route['routing_method']}"
        )

        # Step 2: Planning Task (Concurrent)
        plan_task = asyncio.create_task(
            self.client.generate(PLAN_SYSTEM_PROMPT, user_query, temperature=0.1, max_tokens=1024)
        )
        plan_res = await plan_task

        # Step 3: Final Guided Answer Generation
        guided_system_prompt = self._build_guided_prompt(route, plan_res["answer"])
        final_res = await self.client.generate(
            guided_system_prompt, user_query, temperature=0.1, max_tokens=4096
        )

        return {
            "query": user_query,
            "route": route["tier"],
            "expert_id": route.get("expert_id"),
            "expert_conf": route.get("expert_conf"),
            "margin": route.get("margin"),
            "variant_id": route.get("variant_id"),
            "variant_conf": route.get("variant_conf"),
            "routing_method": route.get("routing_method"),
            "reranked": route.get("reranked", False),
            "cross_encoder_score": route.get("cross_encoder_score"),
            "plan_res": plan_res,
            "final_res": final_res,
        }

    async def shutdown(self):
        await self.client.close()


# ==============================================================================
# 6. RESILIENT CLI ENTRYPOINT
# ==============================================================================
async def main():
    app = CyberMoEOrchestrator()

    print("\n" + "=" * 70)
    print("CYBER MoE ORCHESTRATOR - Plan -> Route -> Guide -> Answer (Dual Embedding Ensemble)")
    print(f"Target Server: {SERVER_URL} | Model: {MODEL_NAME}")
    print("Enter your prompt (or type 'exit' to quit):")
    print("=" * 70 + "\n")

    try:
        while True:
            try:
                user_input = input("\nPrompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting orchestrator. Goodbye!")
                break

            res = await app.process_query(user_input)

            print("\n" + "=" * 70)
            print(f"[Route]   : {res['route']}")
            print(f"[Method]  : {res.get('routing_method', 'unknown')}")
            if res["expert_id"]:
                print(f"[Expert]  : {res['expert_id']} ({EXPERTS[res['expert_id']]['domain']})")
                print(f"[Conf]    : {res['expert_conf']:.3f}")
                print(f"[Margin]  : {res['margin']:.3f}")
            if res.get("reranked"):
                print(f"[Reranked]: YES (CE score={res['cross_encoder_score']:.3f})")
            if res.get("variant_id"):
                v = VARIANTS[res["variant_id"]]
                print(f"[Variant] : {res['variant_id']} ({v['task']}) conf={res['variant_conf']:.2f}")
            print("=" * 70)

            p = res["plan_res"]
            print(f"\n[Plan] ({p['elapsed']:.2f}s):\n{p['answer']}")

            f = res["final_res"]
            tps_str = f" | {f['tokens']} tokens @ {f['tps']:.1f} tok/s" if f["tps"] > 0 else ""
            if f["reasoning"]:
                print(f"\n[Thinking] ({f['elapsed']:.2f}s{tps_str}):\n{f['reasoning']}")
            print(f"\n[Final Answer] ({f['elapsed']:.2f}s{tps_str}):\n{f['answer']}")
            print("=" * 70)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
