# multi_stage_agents_fixed.py
"""
Multi-stage Autogen-style multi-agent pipeline (single file) — FIXED with ReasoningGate

- Stages (in order): Diagnostic -> Predictive -> Prescriptive -> Descriptive
- Uses ReasoningGate (LLM reasoning) to select domains rather than running all.
- Quick-pass (fast partial insight) + full-pass (refinement) per stage,
  incremental re-synthesis as full results arrive, and simple in-memory caching.
- Preserves agent backstories, role prompts, and compression helpers from your original files.
"""

import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# ---------------------- Configuration (tune these) ----------------------
LLM_MODEL = "phi3:mini"
LLM_BASE_URL = "http://localhost:11434"

# Timeouts (increase for local models if needed)
LLM_REQUEST_TIMEOUT = 600
DOMAIN_CALL_TIMEOUT = 300
SYNTHESIS_TIMEOUT = 600

# Quick-pass (fast partial insight)
QUICK_PASS_TIMEOUT = 4
QUICK_PASS_MAX_TOKENS = 120

# Full-pass (refined outputs)
FULL_PASS_MAX_TOKENS = 800

# Parallelism
MAX_WORKERS = 1

COMPRESS_MAX_TOKENS = 250
SYNTH_MAX_TOKENS = 1000

# RAG: toggle and placeholder links/snippets
RAG_ENABLED = True
RAG_LINKS_PLACEHOLDER: List[str] = []

# Safety keywords
SAFETY_KEYWORDS = ["live wire", "electrocution", "shock", "sparking", "arc", "fire", "smoke"]

# ---------------------- Simple in-memory cache (per run) ----------------------
_CALL_CACHE: Dict[Tuple[str, str, int], str] = {}

def _cache_get(agent_name: str, input_text: str, max_tokens: int) -> Optional[str]:
    key = (agent_name, str(hash(input_text)), max_tokens)
    return _CALL_CACHE.get(key)

def _cache_set(agent_name: str, input_text: str, max_tokens: int, output: str):
    key = (agent_name, str(hash(input_text)), max_tokens)
    _CALL_CACHE[key] = output

# ---------------------- Ollama client (shared) ----------------------
try:
    from ollama import Ollama  # type: ignore
    _sdk = True
except Exception:
    import requests  # type: ignore
    _sdk = False

class OllamaClient:
    def __init__(self, model: str = LLM_MODEL, base_url: str = LLM_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        if _sdk:
            self.client = Ollama()

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, dict):
            if "message" in response and isinstance(response["message"], dict) and "content" in response["message"]:
                return response["message"]["content"]
            if "choices" in response and len(response["choices"]) > 0:
                c = response["choices"][0]
                if isinstance(c, dict) and "message" in c and "content" in c["message"]:
                    return c["message"]["content"]
                if isinstance(c, dict) and "text" in c:
                    return c["text"]
            if "text" in response:
                return response["text"]
        return str(response)

    def generate(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
        if _sdk:
            r = self.client.chat(model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens)
            return self._extract_text(r)
        else:
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": temperature, "num_predict": max_tokens}
            }
            res = requests.post(url, json=payload, timeout=LLM_REQUEST_TIMEOUT)
            res.raise_for_status()
            return self._extract_text(res.json())

# ---------------------- Core Agent classes (shared) ----------------------
class Agent:
    def __init__(self, name: str, llm: OllamaClient, role: str, goal: str, backstory: str, verbose: int = 0):
        self.name = name
        self.llm = llm
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.verbose = verbose

    def run(self, instruction: str, max_tokens: int = 600) -> str:
        prompt = f"Role: {self.role}\nGoal: {self.goal}\nBackstory: {self.backstory}\n\nInstruction:\n{instruction}\n"
        if self.verbose:
            print(f"[{self.name}] prompt sent")
        out = self.llm.generate(prompt, max_tokens=max_tokens)
        if self.verbose:
            print(f"[{self.name}] received {len(out)} chars")
        return out

class DomainRegistry:
    def __init__(self):
        self.domains: Dict[str, Agent] = {}

    def register(self, domain: str, agent: Agent):
        self.domains[domain] = agent

    def list_domains(self) -> List[str]:
        return list(self.domains.keys())

    def get(self, domain: str) -> Agent:
        return self.domains[domain]

class RouterAgent(Agent):
    def _keyword_fallback(self, text: str, available_domains: List[str]) -> List[str]:
        # basic keyword fallback if LLM completely fails
        t = text.lower()
        picks = []
        if any(k in t for k in ["motor", "bearing", "vibration", "alignment", "coupling"]):
            if "Mechanical Engineering" in available_domains:
                picks.append("Mechanical Engineering")
        if any(k in t for k in ["relay", "ground", "vfd", "power", "thermography", "wire"]):
            if "Electrical Engineering" in available_domains:
                picks.append("Electrical Engineering")
        if any(k in t for k in ["ph", "reagent", "slurry", "flotation", "assay"]):
            if "Chemical Engineering" in available_domains:
                picks.append("Chemical Engineering")
        return picks if picks else []

    def select_domains(self, problem: str, available_domains: List[str]) -> List[str]:
        instruction = (
            "Given the problem description below, select which domains are REQUIRED from the following list. "
            "Return only a JSON list of domain names.\n\n"
            f"Available domains: {available_domains}\n\nProblem:\n{problem}"
        )
        raw = self.run(instruction)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                return [p for p in parsed if p in available_domains]
        except Exception:
            pass
        return self._keyword_fallback(problem, available_domains)

class SynthesizerAgent(Agent):
    def run(self, instruction: str, max_tokens: int = SYNTH_MAX_TOKENS) -> str:
        return super().run(instruction, max_tokens=max_tokens)

# ---------------------- Reasoning Gate (new) ----------------------
class ReasoningGate(RouterAgent):
    """
    ReasoningGate uses the LLM to *reason* (not keyword-match) which domains are necessary.
    It returns a JSON structure containing selected_domains and short reasoning.
    Robust parsing and fallbacks are provided.
    """
    def select_domains(self, problem: str, available_domains: List[str]) -> List[str]:
        instruction = f"""
You are a REASONING GATE, not a classifier.

Your task is to decide which domains are NECESSARY to solve the problem.

You must reason using these lenses:
1. What is the PRIMARY phenomenon or hazard?
2. What domains have DIRECT causal authority over it?
3. What domains are ONLY secondary or downstream?
4. Which domains can be EXCLUDED safely?

Rules:
- Do NOT select a domain unless it adds unique causal insight.
- Penalize over-selection.
- Multiple domains are allowed ONLY if justified.
- Think abstractly — not by keywords.

Return STRICT JSON ONLY:

{{
  "selected_domains": ["Domain A", "Domain B"],
  "confidence": 0-100,
  "reasoning": {{
    "Domain A": "why it is required",
    "Domain B": "why it is required"
  }},
  "excluded_domains": {{
    "Domain X": "why not required",
    "Domain Y": "why not required"
  }}
}}

Available domains:
{available_domains}

Problem:
{problem}
"""
        raw = self.run(instruction, max_tokens=400)
        # Try robust JSON extraction
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            # attempt to find first {...} block
            try:
                start = raw.index("{")
                end = raw.rindex("}") + 1
                parsed = json.loads(raw[start:end])
            except Exception:
                parsed = None

        if isinstance(parsed, dict):
            sel = parsed.get("selected_domains", [])
            if isinstance(sel, list):
                # filter to available domains, preserve order
                filtered = [d for d in sel if d in available_domains]
                if filtered:
                    return filtered
        # fallback to lightweight keyword-based heuristic (but *not* "select all")
        fallback = super()._keyword_fallback(problem, available_domains)
        if fallback:
            return fallback
        # final safe default: return empty list so caller can decide (fail-closed)
        return []

# ---------------------- RAG placeholders ----------------------
def rag_retrieve_stub(query: str) -> List[str]:
    return RAG_LINKS_PLACEHOLDER.copy()

def rag_summary_for_prompt(docs: List[str]) -> str:
    if not docs:
        return ""
    return "\n\nRAG_DOCUMENTS:\n" + "\n".join(f"- {d}" for d in docs)

# ---------------------- Diagnostic stage ----------------------
def make_diagnostic_domain_agent(domain_name: str, goal: str, problems: str, tools: str, llm: OllamaClient) -> Agent:
    backstory = f"""
You are a {domain_name} Diagnostic Specialist for a gold mining company.
Objective: {goal}

Rules:
- Diagnose only within your domain
- Use evidence-based reasoning
- Provide confidence score and missing data

Typical problems:
{problems}

Tools and data:
{tools}
"""
    return Agent(
        name=domain_name,
        llm=llm,
        role=f"{domain_name} Diagnostic Specialist",
        goal=goal,
        backstory=backstory,
        verbose=1
    )

def compress_diagnostic_output(llm: OllamaClient, domain_name: str, text: str) -> str:
    prompt = (
        f"You are a domain summarizer for {domain_name}.\n\n"
        "Input: a detailed diagnostic report. Produce a JSON object with keys:\n"
        "summary (1-2 sentences), top_hypotheses (list of up to 3 short items), confidence (0-100), recommended_actions (list of up to 3 short items), missing_evidence (list).\n\n"
        "Provide valid JSON only.\n\n"
        f"Report:\n{text}"
    )
    try:
        raw = llm.generate(prompt, max_tokens=COMPRESS_MAX_TOKENS)
        parsed = json.loads(raw)
        return json.dumps(parsed)
    except Exception:
        short = text.strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400] + '...'
        return json.dumps({"summary": short, "top_hypotheses": [], "confidence": 50, "recommended_actions": [], "missing_evidence": []})

# ---------------------- Predictive stage ----------------------
def make_predictive_domain_agent(domain_name: str, short_goal: str, domain_scope: str, key_tools: str, llm: OllamaClient) -> Agent:
    role = f"{domain_name} Predictive Specialist"
    goal = short_goal
    backstory = f"""
You are a {domain_name} Predictive Analytics Expert for a gold mining company.

Objective: {short_goal}

You specialize in forecasting failure progression, operational risk, and future outcomes.
You use trend extrapolation, causal reasoning, scenario simulation, and uncertainty quantification.
Prioritize stating assumptions, confidence intervals, and data gaps.
"""
    class PredictiveStage(Agent):
        def run_stage(self, diagnostic_text: str, max_tokens: int = FULL_PASS_MAX_TOKENS) -> str:
            instruction = (
                "Given the diagnostic findings or problem description below, produce a structured prediction. "
                "For each scenario (best / base / worst) include: brief description, likely timeline, likelihood (0-100), key drivers, and confidence. "
                "Also list explicit assumptions and missing evidence that would materially change the forecast.\n\n"
                f"Input:\n{diagnostic_text}"
            )
            return self.run(instruction, max_tokens=max_tokens)
    return PredictiveStage(name=f"Predictive - {domain_name}", llm=llm, role=role, goal=goal, backstory=backstory, verbose=1)

def compress_predictive_output(llm: OllamaClient, domain_name: str, text: str) -> str:
    prompt = (
        f"You are a domain summarizer for {domain_name}.\n\n"
        "Input: a detailed predictive or diagnostic report. Produce a JSON object with keys:\n"
        "summary (1-2 sentences), top_predictions (list up to 3 with timeline & likelihood), confidence (0-100), leading_indicators (list), missing_evidence (list).\n\n"
        "Provide valid JSON only.\n\n"
        f"Report:\n{text}"
    )
    try:
        raw = llm.generate(prompt, max_tokens=COMPRESS_MAX_TOKENS)
        parsed = json.loads(raw)
        return json.dumps(parsed)
    except Exception:
        short = text.strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400] + '...'
        return json.dumps({"summary": short, "top_predictions": [], "confidence": 50, "leading_indicators": [], "missing_evidence": []})

# ---------------------- Prescriptive stage ----------------------
def make_prescriptive_domain_agent(domain_name: str, short_goal: str, domain_scope: str, key_tools: str, llm: OllamaClient) -> Agent:
    role = f"{domain_name} Prescriptive Specialist"
    goal = short_goal
    backstory = (
        f"You are a {domain_name} Prescriptive AI for a gold mining company.\n\n"
        f"Objective: {short_goal}\n\n"
        "You specialize in decision intelligence: producing ranked options, solutions, choices, "
        "decisions, and concrete actions with implementation options and monitoring plans. "
        "Prioritize safety, regulatory compliance, and explicit assumptions.\n"
    )
    class PrescriptiveStage(Agent):
        def run_stage(self, problem_text: str, max_tokens: int = FULL_PASS_MAX_TOKENS) -> str:
            instruction = (
                "Given the problem description below, produce a structured prescriptive report. "
                "For each ranked option include: title (Recommended/Alternative/Conservative), expected outcome, confidence (0-100), "
                "three actionable sub-steps (who/what/when), estimated cost/time, required approvals, monitoring & rollback plans, risks, dependencies, and explicit assumptions. "
                "Also provide a short decision summary listing the recommended decision, alternative choices, and implementation actions/options.\n\n"
                f"Problem:\n{problem_text}"
            )
            return self.run(instruction, max_tokens=max_tokens)
    return PrescriptiveStage(name=f"Prescriptive - {domain_name}", llm=llm, role=role, goal=goal, backstory=backstory, verbose=1)

def compress_prescriptive_output(llm: OllamaClient, domain_name: str, text: str) -> str:
    prompt = (
        f"You are a prescriptive domain summarizer for {domain_name}.\n\n"
        "Input: a detailed prescriptive report. Produce a JSON object with keys:\n"
        "summary (1-2 sentences), top_options (list up to 3 with title, short rationale, confidence), "
        "decision_summary (one-line recommended decision), assumptions (list), required_data (list).\n\n"
        "Provide valid JSON only.\n\n"
        f"Report:\n{text}"
    )
    try:
        raw = llm.generate(prompt, max_tokens=COMPRESS_MAX_TOKENS)
        parsed = json.loads(raw)
        return json.dumps(parsed)
    except Exception:
        short = text.strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400] + '...'
        return json.dumps({"summary": short, "top_options": [], "decision_summary": "", "assumptions": [], "required_data": []})

# ---------------------- Descriptive stage ----------------------
def make_descriptive_domain_agent(domain_name: str, short_goal: str, domain_scope: str, key_tools: str, llm: OllamaClient) -> Agent:
    role = f"{domain_name} Descriptive Agent"
    goal = short_goal
    backstory = (
        f"You are a {domain_name} Descriptive Analytics Expert for a gold mining company.\n\n"
        f"Objective: {short_goal}\n\n"
        "Backstory: You are an experienced field specialist who summarizes what has happened, what is happening now, "
        "and what is likely to happen next in your domain. You extract trends, patterns, anomalies, and actionable insights "
        "from historical and real-time data. Prioritize clarity, evidence, and traceability.\n"
    )
    class DescriptiveStage(Agent):
        def run_stage(self, input_text: str, max_tokens: int = FULL_PASS_MAX_TOKENS) -> str:
            instruction = (
                "Given the dataset description or diagnostic/problem text below, produce a structured descriptive report. "
                "Follow the Structured output format in the backstory.\n\n"
                f"Input:\n{input_text}"
            )
            return self.run(instruction, max_tokens=max_tokens)
    return DescriptiveStage(name=f"Descriptive - {domain_name}", llm=llm, role=role, goal=goal, backstory=backstory, verbose=1)

def compress_descriptive_output(llm: OllamaClient, domain_name: str, text: str) -> str:
    prompt = (
        f"You are a descriptive domain summarizer for {domain_name}.\n\n"
        "Input: a detailed descriptive report. Produce a JSON object with keys:\n"
        "summary (1-2 sentences), key_metrics (list), notable_anomalies (list), short_term_outlook (1-2 lines with confidence), audit_log (list).\n\n"
        "Provide valid JSON only.\n\n"
        f"Report:\n{text}"
    )
    try:
        raw = llm.generate(prompt, max_tokens=COMPRESS_MAX_TOKENS)
        parsed = json.loads(raw)
        return json.dumps(parsed)
    except Exception:
        short = text.strip().replace("\n", " ")
        if len(short) > 400:
            short = short[:400] + '...'
        return json.dumps({"summary": short, "key_metrics": [], "notable_anomalies": [], "short_term_outlook": "", "audit_log": []})

# ---------------------- Utility: safe domain call with timeout and token control ----------------------
def call_domain_agent(agent: Agent, domain_input: str, timeout: int = DOMAIN_CALL_TIMEOUT, max_tokens: int = FULL_PASS_MAX_TOKENS) -> str:
    cached = _cache_get(agent.name, domain_input, max_tokens)
    if cached is not None:
        return cached

    def _call():
        try:
            if hasattr(agent, "run_stage"):
                try:
                    return agent.run_stage(domain_input, max_tokens=max_tokens)
                except TypeError:
                    return agent.run(domain_input, max_tokens=max_tokens)
            else:
                return agent.run(domain_input, max_tokens=max_tokens)
        except Exception as e:
            return f"[{agent.name}] ERROR: {e}\n{traceback.format_exc()}"

    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_call)
        try:
            out = future.result(timeout=timeout)
            _cache_set(agent.name, domain_input, max_tokens, out)
            return out
        except Exception as e:
            future.cancel()
            return f"[{agent.name}] TIMEOUT or ERROR: {e}"

# ---------------------- Stage runner ----------------------
def run_stage(
    stage_name: str,
    router: RouterAgent,
    registry: DomainRegistry,
    synthesizer: SynthesizerAgent,
    compress_fn,
    upstream_text: str,
    llm_client: OllamaClient,
    rag_enabled: bool = RAG_ENABLED,
    allowed_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    available_domains = registry.list_domains()

    if allowed_domains is not None:
        selected = [d for d in allowed_domains if d in available_domains]
    else:
        try:
            selected = router.select_domains(upstream_text, available_domains)
        except Exception:
            selected = []

    # If gate returned nothing, fail-closed to a safe minimal default (first domain)
    if not selected:
        if available_domains:
            print(f"[{stage_name}] Reasoning gate returned no domains. Using safe default: [{available_domains[0]}]")
            selected = [available_domains[0]]
        else:
            selected = []

    domain_outputs: Dict[str, str] = {}
    compressed_outputs: Dict[str, str] = {}

    rag_docs = rag_retrieve_stub(upstream_text) if rag_enabled else []
    rag_text = rag_summary_for_prompt(rag_docs)

    domain_inputs = {d: upstream_text + ("\n\n" + rag_text if rag_text else "") for d in selected}

    # QUICK PASS
    print(f"[{stage_name}] quick-pass: requesting fast partial results (timeout {QUICK_PASS_TIMEOUT}s)...")
    quick_futures = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(selected)))) as ex:
        for d in selected:
            agent = registry.get(d)
            quick_futures[ex.submit(call_domain_agent, agent, domain_inputs[d], QUICK_PASS_TIMEOUT, QUICK_PASS_MAX_TOKENS)] = d
        for fut in as_completed(quick_futures):
            d = quick_futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                out = f"[{d}] ERROR during quick-pass: {e}"
            domain_outputs[d] = out
            try:
                comp = compress_fn(llm_client, d, out)
            except Exception:
                comp = json.dumps({"summary": out.strip()[:400]})
            compressed_outputs[d] = comp

    quick_synth_input = "\n\n".join(f"[{d}]\n{compressed_outputs[d]}" for d in selected)
    quick_stage_report = ""
    try:
        quick_stage_report = synthesizer.run(quick_synth_input, max_tokens=min(SYNTH_MAX_TOKENS, 400))
    except Exception:
        quick_stage_report = quick_synth_input

    print(f"\n[{stage_name}] QUICK STAGE REPORT (partial):\n")
    print(quick_stage_report)
    print("\n--- End quick partial report ---\n")

    # FULL PASS
    print(f"[{stage_name}] full-pass: requesting refined results (timeout {DOMAIN_CALL_TIMEOUT}s)...")
    full_futures = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(1, len(selected)))) as ex:
        for d in selected:
            agent = registry.get(d)
            full_futures[ex.submit(call_domain_agent, agent, domain_inputs[d], DOMAIN_CALL_TIMEOUT, FULL_PASS_MAX_TOKENS)] = d

        for fut in as_completed(full_futures):
            d = full_futures[fut]
            try:
                out = fut.result()
            except Exception as e:
                out = f"[{d}] ERROR during full-pass: {e}"
            domain_outputs[d] = out
            try:
                comp = compress_fn(llm_client, d, out)
            except Exception:
                comp = json.dumps({"summary": out.strip()[:400]})
            compressed_outputs[d] = comp

            synth_input = "\n\n".join(f"[{dd}]\n{compressed_outputs[dd]}" for dd in selected)
            stage_report = ""
            try:
                stage_report = synthesizer.run(synth_input, max_tokens=SYNTH_MAX_TOKENS)
            except Exception:
                stage_report = synth_input

            print(f"\n[{stage_name}] incremental update after '{d}' finished:\n")
            print(stage_report)
            print("\n--- End incremental update ---\n")

    final_synth_input = "\n\n".join(f"[{d}]\n{compressed_outputs[d]}" for d in selected)
    final_stage_report = ""
    try:
        final_stage_report = synthesizer.run(final_synth_input, max_tokens=SYNTH_MAX_TOKENS)
    except Exception:
        final_stage_report = final_synth_input

    return {
        "selected_domains": selected,
        "domain_outputs": domain_outputs,
        "compressed_outputs": compressed_outputs,
        "stage_report": final_stage_report
    }

# ---------------------- Build registries and routers for each stage ----------------------
def build_diagnostic_stage(llm: OllamaClient):
    router = ReasoningGate(
        name="diag_router", llm=llm, role="Reasoning Gate (Diagnostic)",
        goal="Identify which engineering domains are required to solve a problem",
        backstory="You analyze mining equipment problems and route them to the correct specialists.",
        verbose=0
    )
    synthesizer = SynthesizerAgent(
        name="diag_writer", llm=llm, role="Diagnostic Synthesizer",
        goal="Synthesize diagnostic domain outputs", backstory="Synthesize diagnostic outputs.",
        verbose=0
    )
    registry = DomainRegistry()

    registry.register("Mechanical Engineering", make_diagnostic_domain_agent(
        "Mechanical Engineering",
        "Diagnose mechanical faults in mining machinery",
        "Vibration, misalignment, bearing wear, overheating",
        "Vibration data, maintenance logs, alignment checks",
        llm
    ))
    registry.register("Electrical Engineering", make_diagnostic_domain_agent(
        "Electrical Engineering",
        "Diagnose electrical and power faults",
        "Breaker trips, grounding issues, motor faults",
        "Electrical schematics, PLC logs, thermography",
        llm
    ))
    registry.register("Chemical Engineering", make_diagnostic_domain_agent(
        "Chemical Engineering",
        "Diagnose process and reagent issues",
        "pH imbalance, reagent dosing, slurry issues",
        "Process data, lab assays, mass balance",
        llm
    ))
    return router, registry, synthesizer

def build_predictive_stage(llm: OllamaClient):
    router = ReasoningGate(
        name="pred_router", llm=llm, role="Reasoning Gate (Predictive)",
        goal="Identify which domains are required to forecast outcomes",
        backstory="Route to predictive specialists.", verbose=0
    )
    synthesizer = SynthesizerAgent(
        name="pred_writer", llm=llm, role="Predictive Synthesizer",
        goal="Synthesize predictive domain outputs", backstory="Synthesize predictive outputs.",
        verbose=0
    )
    registry = DomainRegistry()
    registry.register("Mechanical Engineering", make_predictive_domain_agent(
        "Mechanical Engineering",
        "Forecast mechanical failure progression and likelihoods",
        "bearing degradation, vibration escalation, gearbox wear, alignment drift",
        "vibration trend data, historical failure records, thermal logs",
        llm
    ))
    registry.register("Electrical Engineering", make_predictive_domain_agent(
        "Electrical Engineering",
        "Forecast electrical component degradation and failure probabilities",
        "motor insulation breakdown, breaker trip trends, grounding deterioration",
        "insulation resistance logs, PLC event history, thermographic trends",
        llm
    ))
    registry.register("Chemical Engineering", make_predictive_domain_agent(
        "Chemical Engineering",
        "Forecast process drift and reagent-related performance risks",
        "pH drift, reagent depletion, flotation response degradation",
        "process trends, assay history, dosing logs",
        llm
    ))
    return router, registry, synthesizer

def build_prescriptive_stage(llm: OllamaClient):
    router = ReasoningGate(
        name="pres_router", llm=llm, role="Reasoning Gate (Prescriptive)",
        goal="Identify which domains are required to prescribe actions",
        backstory="Route to prescriptive specialists.", verbose=0
    )
    synthesizer = SynthesizerAgent(
        name="pres_writer", llm=llm, role="Prescriptive Synthesizer",
        goal="Synthesize prescriptive domain outputs", backstory="Synthesize prescriptive outputs.",
        verbose=0
    )
    registry = DomainRegistry()
    registry.register("Mechanical Engineering", make_prescriptive_domain_agent(
        "Mechanical Engineering",
        "Prescribe actions to reduce mechanical downtime",
        "Alignment corrections, bearing replacement, lubrication program, spare parts strategy",
        "CMMS, OEM manuals, reliability models",
        llm
    ))
    registry.register("Electrical Engineering", make_prescriptive_domain_agent(
        "Electrical Engineering",
        "Prescribe power system risk reductions",
        "Relay coordination, grounding improvements, motor protection, UPS strategy",
        "Protection studies, electrical schematics, thermography",
        llm
    ))
    registry.register("Chemical Engineering", make_prescriptive_domain_agent(
        "Chemical Engineering",
        "Prescribe process optimization and reagent strategy",
        "Reagent tuning, temperature control, residence time adjustments",
        "Process simulators, assay history, dosing logs",
        llm
    ))
    return router, registry, synthesizer

def build_descriptive_stage(llm: OllamaClient):
    router = ReasoningGate(
        name="desc_router", llm=llm, role="Reasoning Gate (Descriptive)",
        goal="Identify which domains are required to describe trends and current state",
        backstory="Route to descriptive specialists.", verbose=0
    )
    synthesizer = SynthesizerAgent(
        name="desc_writer", llm=llm, role="Descriptive Synthesizer",
        goal="Synthesize descriptive domain outputs", backstory="Synthesize descriptive outputs.",
        verbose=0
    )
    registry = DomainRegistry()
    registry.register("Mechanical Engineering", make_descriptive_domain_agent(
        "Mechanical Engineering",
        "Describe mechanical system health, failures, and near-term risks",
        "Alignment, bearing condition, lubrication trends, vibration escalation",
        "CMMS history, vibration analytics, thermal imaging",
        llm
    ))
    registry.register("Electrical Engineering", make_descriptive_domain_agent(
        "Electrical Engineering",
        "Describe electrical system performance and emerging risks",
        "Relay events, grounding integrity, load imbalance trends",
        "SCADA logs, protection events, thermography",
        llm
    ))
    registry.register("Chemical Engineering", make_descriptive_domain_agent(
        "Chemical Engineering",
        "Describe process stability, reagent usage trends, and recovery indicators",
        "pH trends, reagent consumption, slurry density stability",
        "Process historian, lab assays, mass balance logs",
        llm
    ))
    return router, registry, synthesizer

# ---------------------- Helpers ----------------------
def ollama_health_check(base_url: str) -> bool:
    if _sdk:
        return True
    try:
        import requests
        r = requests.get(base_url + "/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def is_immediate_hazard(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in SAFETY_KEYWORDS)

def run_pipeline(problem: str) -> str:
    # everything you currently do after user input
    return final_output

# ---------------------- Main pipeline ----------------------
def main():
    print("Multi-stage agents pipeline starting...")
    llm = OllamaClient(model=LLM_MODEL, base_url=LLM_BASE_URL)
    if not ollama_health_check(LLM_BASE_URL):
        print(f"Ollama model not reachable at {LLM_BASE_URL}. Please start the model or adjust LLM_BASE_URL and try again.")
        return

    diag_router, diag_registry, diag_synth = build_diagnostic_stage(llm)
    pred_router, pred_registry, pred_synth = build_predictive_stage(llm)
    pres_router, pres_registry, pres_synth = build_prescriptive_stage(llm)
    desc_router, desc_registry, desc_synth = build_descriptive_stage(llm)

    if RAG_ENABLED:
        print("RAG is enabled. If you have short grounding snippets or links to include, paste them now (one per line).")
        print("When done, enter an empty line. Or just press Enter to skip.")
        rag_lines = []
        while True:
            try:
                line = input().strip()
            except EOFError:
                line = ""
            if not line:
                break
            rag_lines.append(line)
        if rag_lines:
            global RAG_LINKS_PLACEHOLDER
            RAG_LINKS_PLACEHOLDER = rag_lines

    user_input = input("Describe the problem or paste diagnostic output: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        return

    if is_immediate_hazard(user_input):
        print("\nImmediate safety notice: This sounds like an electrical or fire hazard.")
        print("Stop work, isolate power if safe to do so, call qualified personnel, and follow lockout/tagout procedures.")
        confirm = input("Do you want to continue and run the full diagnostic pipeline (non-emergency)? (yes/no): ").strip().lower()
        if confirm not in ("yes", "y"):
            print("Aborting pipeline due to safety short-circuit. Stay safe.")
            return

    aggregated: Dict[str, Dict[str, Any]] = {}

    # Diagnostic
    print("\n--- Running Diagnostic Stage ---\n")
    diag_result = run_stage(
        stage_name="Diagnostic",
        router=diag_router,
        registry=diag_registry,
        synthesizer=diag_synth,
        compress_fn=compress_diagnostic_output,
        upstream_text=user_input,
        llm_client=llm,
        rag_enabled=RAG_ENABLED,
        allowed_domains=None
    )
    aggregated["Diagnostic"] = diag_result
    diagnostic_selected = diag_result["selected_domains"]
    diagnostic_report = diag_result["stage_report"]

    # Predictive (inherit diagnostic_selected)
    print("\n--- Running Predictive Stage ---\n")
    pred_input = diagnostic_report if diagnostic_report else user_input
    pred_result = run_stage(
        stage_name="Predictive",
        router=pred_router,
        registry=pred_registry,
        synthesizer=pred_synth,
        compress_fn=compress_predictive_output,
        upstream_text=pred_input,
        llm_client=llm,
        rag_enabled=RAG_ENABLED,
        allowed_domains=diagnostic_selected
    )
    aggregated["Predictive"] = pred_result
    predictive_selected = pred_result["selected_domains"]
    predictive_report = pred_result["stage_report"]

    # Prescriptive (inherit predictive_selected)
    print("\n--- Running Prescriptive Stage ---\n")
    pres_input = predictive_report if predictive_report else pred_input
    pres_result = run_stage(
        stage_name="Prescriptive",
        router=pres_router,
        registry=pres_registry,
        synthesizer=pres_synth,
        compress_fn=compress_prescriptive_output,
        upstream_text=pres_input,
        llm_client=llm,
        rag_enabled=RAG_ENABLED,
        allowed_domains=predictive_selected
    )
    aggregated["Prescriptive"] = pres_result
    prescriptive_selected = pres_result["selected_domains"]
    prescriptive_report = pres_result["stage_report"]

    # Descriptive (inherit prescriptive_selected)
    print("\n--- Running Descriptive Stage ---\n")
    desc_input = prescriptive_report if prescriptive_report else pres_input
    desc_result = run_stage(
        stage_name="Descriptive",
        router=desc_router,
        registry=desc_registry,
        synthesizer=desc_synth,
        compress_fn=compress_descriptive_output,
        upstream_text=desc_input,
        llm_client=llm,
        rag_enabled=RAG_ENABLED,
        allowed_domains=prescriptive_selected
    )
    aggregated["Descriptive"] = desc_result

    # Print aggregated results
    print("\n\n================ AGGREGATED PIPELINE OUTPUT ================\n")
    for stage in ["Diagnostic", "Predictive", "Prescriptive", "Descriptive"]:
        res = aggregated.get(stage, {})
        print(f"--- {stage} STAGE ---\n")
        print("Selected domains:", res.get("selected_domains", []))
        print("\n[Raw domain outputs]\n")
        for d, out in (res.get("domain_outputs") or {}).items():
            print(f"[{d}]")
            print(out)
            print("-" * 60)
        print("\n[Compressed summaries]\n")
        for d, comp in (res.get("compressed_outputs") or {}).items():
            print(f"[{d}] {comp}")
        print("\n[Stage synthesis]\n")
        print(res.get("stage_report", ""))
        print("\n" + "=" * 70 + "\n")

    print("\n================ END OF PIPELINE ================\n")

if __name__ == "__main__":
    main()
