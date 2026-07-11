# GLUE-fw Deep Analysis: StickyScript, MCP Integration, and What's Worth Keeping

*An evidence-based audit of the framework as of `main` (8a0f653, May 2025), performed by
reading the full source, running the test suite, and executing the framework end-to-end
with instrumented providers. Every claim below carries a file:line reference or a
reproducible experiment.*

---

## 1. What was actually run

| Experiment | Result |
|---|---|
| `pytest tests/` (38 tests) | **All pass** — but they cover only JSON schemas, stubbed agent-loop phases, and retry logic. No test touches the DSL parser, providers, adhesives, flows, or MCP. |
| Parse all 6 shipped `examples/*.glue` | **2 of 6 fail with `SyntaxError`** (`research_assistant.glue`, `multi_agent_team.glue`) |
| `glue run` CLI | **Broken at import time**, two ways (see §2.4) |
| Full end-to-end orchestration with a scripted fake LLM provider | Lead decomposes goal → delegates → member runs its loop → reports success → **then the system deadlocks permanently** (see §3.3) |

## 2. The StickyScript DSL: how agentic pipelines are actually built

### 2.1 The pipeline

The DSL is a hand-written lexer (`src/glue/dsl/lexer.py`, ~170 lines, regex token table)
and recursive-descent parser (`src/glue/dsl/parser.py`, ~615 lines). It is a **static
configuration language, not an expression language** — no variables, no expressions, no
conditionals, no interpolation. It parses into a plain dict
(`{"app", "models", "tools", "magnetize", "flows", "apply"}`; `parser.py:29-37`), not an AST.

At runtime: `.glue` file → tokens → config dict → `GlueApp(config=ast)` (`cli.py:466`) →
teams built from the `magnetize` block, one `TeamLeadAgentLoop` per team lead and one
`TeamMemberAgentLoop` per member (`teams.py:690-729`). Orchestration is:
lead LLM decomposes the goal into subtasks (JSON) → `delegate_task` writes tasks into
`team.shared_results` → members poll that dict every 0.5s (`teams.py:1520-1531`) →
members run a phase machine (parse → plan → select tool → self-evaluate → report) →
lead polls for `"completion"` records → retries failures with a different agent →
synthesizes a final JSON answer (`agent_loop.py:718-1226`).

Notably, **tool invocation is prompt-engineered JSON, not native tool-calling**, in every
agent-loop phase: each phase asks the model for fenced JSON and validates with Pydantic
(`agent_loop.py:354-658`). Native provider tool-calls exist only in the separate
`BaseModel.generate` path (`base_model.py:274-351`) used for interactive chat.

### 2.2 The parser rejects the documented syntax

The README's flagship example writes flows directly inside `magnetize`:

```glue
magnetize {
    research { lead = researcher }
    docs { lead = writer }
    researchers -> docs
    docs <- pull
}
```

`_parse_magnetize` (`parser.py:156-208`) only accepts team blocks and a `flow {}` section
there — a bare `researchers -> docs` line throws
`SyntaxError: Expected '{' after team name 'researchers'`. This is why
`examples/research_assistant.glue` (the README example) and
`examples/multi_agent_team.glue` **do not parse**. Additional parser defects found:

- `docs <- pull` (inside `flow {}`) parses `pull` as a *target team name*, producing a
  flow `docs → pull` referencing a nonexistent team, rather than "docs may pull".
- String arrays keep their quote characters: `languages = ["python"]` yields
  `['"python"']` (quotes are stripped for scalar strings at `parser.py:408-414` but never
  for array elements).
- Two parallel build paths exist: `GlueApp._setup_from_dict` (live) and
  `GlueAppBuilder` (`dsl/app_builder.py`, dead — zero imports in `src/`, and it calls
  `Team.add_tool(name=, tool=, binding=)` with a keyword the method doesn't accept).

### 2.3 The branded semantics are mostly not implemented

- **Adhesives (GLUE/VELCRO/TAPE)** — the framework's headline idea (tool-result
  persistence scoping) is ~90% ceremonial at runtime. The complete `AdhesiveSystem`
  (`core/adhesive.py`) is instantiated once (`app.py:123`) and never used again. The live
  path, `Team.share_result` (`teams.py:223-260`), picks an adhesive **nondeterministically**
  (`next(iter(model.adhesives))` on a set), implements GLUE as a last-write-wins dict
  write keyed by tool name, and implements VELCRO and TAPE as *log statements only*.
  Nothing ever feeds stored results back into model context.
- **Magnetic field** — the `Flow` class (`core/flow.py`) is real asyncio-queue message
  passing between teams, and works. But the `MagneticField` layer with its congestion
  metrics, health monitoring, and self-healing rerouting (`magnetic/field.py:222-385`)
  operates on a `flows` dict that is never populated (`app.py:485` bypasses
  `field.add_team`/`set_flow`), and would crash if it were: it awaits
  `team.set_relationship(...)`, a method that does not exist on `Team`.

### 2.4 Operational reality

- The installed CLI is **broken on a fresh install**: `glue/dsl/__init__.py` is empty, so
  `from glue.dsl import GlueDSLParser, GlueLexer` (`cli.py:46`) raises `ImportError`; and
  `cli.py:24` imports `rich`, which is not declared in `pyproject.toml` dependencies.
- Production code special-cases unit tests: `core/app.py` imports `unittest.mock` at
  runtime and branches on whether its own models are `MagicMock`s (`app.py:560-591`);
  the sandbox hardcodes the string literals from its own test cases to pass timeout and
  memory-limit tests (`core/sandbox.py:119-150, 198-202`).
- A 598 KB UTF-16 PowerShell debug log (`output.txt`) is committed at the repo root.

### 2.5 Empirical end-to-end run

With all four models redirected to a scripted fake provider (valid JSON for each phase),
on `examples/agent_communication.glue`:

1. Setup works; teams and loops start.
2. **Every member appears twice** in `team.config.members`
   (`['assistant_1','assistant_2','assistant_3','assistant_1','assistant_2','assistant_3']`),
   so two competing loops per member race on the same task queue.
3. The CLI's non-interactive path starts **two lead orchestrators** for the same goal
   (`cli.py:480-483` auto-starts one inside `start_agent_loops`, then `cli.py:499-515`
   builds a second) — every subtask is delegated and executed twice.
4. Subtask 1 completes and is reported. Then the system **deadlocks forever**:
   `TeamMemberAgentLoop.start` calls `self.terminate(...)` after finishing its *first*
   task (`agent_loop.py:280`), killing the whole `while not self.terminated` fetch loop.
   Each member can ever process exactly one task. The lead's report-wait is an infinite
   1-second poll with no timeout (`agent_loop.py:972-980`; the `self.timeout` field is
   never applied), so any goal with more subtasks than living members hangs `glue run`
   permanently. This is not an edge case — it is the *default* behavior of the primary
   workflow.

Also: if a member's LLM output fails to parse, every phase silently falls back to a stub
that sleeps one second and reports success with high confidence
(`agent_loop.py:322-329, 589-591, 651-658`) — so multi-agent runs can "succeed" while
doing nothing.

## 3. The "native MCP integration"

### 3.1 It is not MCP

`glue/core/mcp.py` implements the "**Model Control Protocol**" (`mcp.py:29`) — a bespoke
scheme: HTTP POST of `{"action", "parameters", "metadata"}` to a single endpoint, with
retries. It shares nothing with Anthropic's **Model Context Protocol** (JSON-RPC 2.0,
`initialize` handshake, `tools/list`/`tools/call`, resources, prompts, stdio/HTTP
transports). There is no `mcp` package dependency. It cannot talk to any real MCP server,
and no real MCP client can talk to its servers. The name is a collision, nothing more.

### 3.2 It is also unreachable

- Zero code outside `mcp.py`/`mcp_factory.py` imports `MCPTool`, `MCPServer`, or
  `DynamicMCPFactory` (verified by grep over `src/` and `tests/`).
- The DSL has **no `mcp` keyword** — there is no way to declare an MCP anything in a
  `.glue` file.
- `mcp_factory.register_with_team` calls `team.register_mcp(...)` — a method `Team`
  does not have (would raise `AttributeError`).
- `glue forge mcp` generates a file that does `from glue.core.mcp import BaseMCP`
  (`cli.py:2042`) — **`BaseMCP` does not exist** — then tells the user to add an
  `mcp name { custom = true }` block to their `.glue` file, which the parser rejects
  with a `SyntaxError` (verified).
- The two MCP docs (`docs/MCP_creation.md`, `docs/mcp_creation_explanation.md`) are
  pasted chat transcripts from AI assistants, not documentation of this codebase.

### 3.3 Is it useful? Better ways?

Not useful as-is: it is a dead-code HTTP wrapper with a misleading name, unreachable from
the product surface. The *intent* — agents dynamically creating and serving tools over a
protocol — is legitimate and is exactly what the real MCP ecosystem now provides:

- **Official MCP Python SDK / FastMCP**: a spec-compliant server is ~10 lines
  (`@mcp.tool()` on a function), with stdio and HTTP transports, auth, and an enormous
  ecosystem of interoperable clients (Claude, IDEs, other frameworks) and thousands of
  existing servers.
- The interesting half of GLUE's idea — *sandboxed runtime tool synthesis by agents* —
  is the hard part, and GLUE's version isn't real either: `create_from_code` defaults to
  un-sandboxed `exec()` (`mcp_factory.py:127, 167`), the `CodeSandbox` behind the
  opt-in flag is in-process string-matching (trivially escapable, with its own tests'
  answers hardcoded in), and the natural-language path generates a template whose body
  is a TODO. Anyone wanting this today should generate a FastMCP server file and run it
  in an OS-level sandbox (container/subprocess with rlimits), not in-process `exec`.

Verdict: adopt the official SDK; there is nothing in this MCP layer worth porting.

## 4. What should NOT be abandoned

Ranked by real, defensible value:

### 4.1 The StickyScript surface language and its two ideas (the keeper)

The genuinely original artifact in this repo is not code — it is **design vocabulary**:

1. **Adhesives as persistence-scope semantics for tool results.** GLUE = team-wide
   persistent, VELCRO = session-scoped, TAPE = one-shot. This is a clean, teachable
   answer to a problem the industry now calls *context engineering / memory scoping* —
   deciding which intermediate results enter which agents' context and for how long.
   Mainstream frameworks (LangGraph, CrewAI, AutoGen) express this imperatively and
   verbosely; none has a one-word declarative vocabulary for it. That three-tier scoping
   maps directly onto modern runtimes (e.g., shared state/store vs. thread state vs.
   ephemeral tool output in LangGraph).
2. **Magnetic flow operators as team topology.** `research -> docs`, `docs <- pull`,
   `><`, `<>` is a five-character syntax for inter-team communication topology (push /
   pull / bidirectional / repel) that reads better than any current framework's graph
   wiring code.

A `.glue`-style file is a readable, reviewable, diffable artifact that a non-engineer can
follow — that promise (the "Terraform for agent teams" niche) still has no clear winner
in 2026. The salvage path is **not** to fix this runtime: retarget the ~800-line
lexer/parser as a front-end that compiles to a maintained runtime (LangGraph graphs,
CrewAI crews, or Claude Agent SDK subagent definitions). The parser needs its flow-line
and quoting bugs fixed (§2.2), which is days of work, not months.

### 4.2 The lead-orchestrator state machine as a blueprint

`TeamLeadAgentLoop` (`agent_loop.py:718-1226`) encodes a solid pattern: LLM goal
decomposition into dependency-ordered subtasks → least-busy assignment → Pydantic-validated
phase outputs → **retry-with-a-different-agent** on failure → LLM synthesis gated on at
least one real success. The *shape* (especially schema-validated phase outputs and
agent-diverse retries) is worth carrying into any successor, even though this
implementation has the fatal loop-termination and no-timeout bugs of §2.5.

### 4.3 Graceful tool-calling degradation (period-piece, still clever)

The OpenRouter provider detects "no endpoints support tool use", caches that per model
class-wide, and transparently falls back to prompt-injected JSON simulated tool calls
(`providers/openrouter.py:163-372`), paired with a forgiving JSON extractor
(`utils/json_utils.py`) that repairs fences, braces, trailing commas, and even key typos.
This solved a real 2025 pain point (free-tier models without tool support). Its value is
fading — native tool calling is near-universal and libraries like `json-repair` are
maintained equivalents — but the *capability-detect → cache → degrade* pattern remains a
good design for budget-model agent systems. (If reused, fix: module-level
`logging.basicConfig(DEBUG)` at `json_utils.py:8`, and the typo table rewriting the
legitimate key `"tool_call"` → `"tool_name"` inside otherwise-valid JSON.)

### 4.4 Does the keeper solve a known problem?

Yes, both halves of §4.1 target documented pain in today's ecosystem: multi-agent
context/memory scoping (what results are shared, with whom, for how long) and
declarative, reviewable team topology. The adhesive vocabulary *enhances something that
already exists* (it could be a thin declarative layer over LangGraph/CrewAI/Agent-SDK
memory and wiring) rather than requiring a new runtime — which is exactly why it can
outlive this codebase.

## 5. Not worth keeping

- The MCP layer (all of it — see §3).
- The adhesive/magnetic *runtime* (dead `AdhesiveSystem`, log-only VELCRO/TAPE, crashing
  `MagneticField`).
- The sandbox (`core/sandbox.py`) — in-process `exec` with substring checks and
  hardcoded test answers; a security liability if anyone trusts its name.
- The providers: OpenAI is a pure mock returning `"Mock OpenAI response"`
  (`providers/openai.py:58-60`); Anthropic awaits a synchronous client and cannot
  complete a call (`providers/anthropic.py:44,162`); Portkey wrapping calls methods no
  provider defines.
- `glue forge` — advertises "AI assistance", collects and stores an API key
  (`cli.py:2311-2398`), and never calls an LLM; it writes static TODO templates.

## 6. Bottom line

GLUE-fw is a 2.5-month prototype (Mar–May 2025) whose skeleton — DSL → teams → queues →
lead/member loops over OpenRouter/Gemini — genuinely runs, but whose three headline
features (adhesives, magnetic fields, MCP) are respectively ceremonial, decorative, and
dead code, and whose default orchestration path deadlocks after one task per member. It
should not be revived as a runtime. What deserves to survive is the **language**: the
adhesive persistence-scope vocabulary, the magnetic topology operators, and the
schema-validated orchestration blueprint — ideally as a small compiler from `.glue`
files onto a maintained agent runtime.
