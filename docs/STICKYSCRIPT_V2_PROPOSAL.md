# StickyScript v2: From Abandoned Runtime to Compiler

*A concrete proposal for the salvage path identified in `STICKYSCRIPT_MCP_ANALYSIS.md`.
The thesis in one sentence: stop shipping a runtime, ship a **compiler** — StickyScript
becomes to agent teams what Terraform's HCL is to infrastructure: a small declarative
language with enforced semantics that compiles onto runtimes other people maintain.*

---

## 1. What the deliverable is

Three artifacts, in dependency order:

1. **A language specification** (`SPEC.md`, ~15 pages). The product is the *semantics*,
   not the syntax: precise definitions of what each adhesive guarantees, what each flow
   operator guarantees, and what a conforming backend must do. This is what GLUE-fw
   never had — its adhesives meant whatever `teams.py` happened to do (mostly nothing).
2. **`stickyc`** — a Python package (PyPI) containing the parser (salvaged from GLUE-fw,
   ~800 lines, bugs fixed), a typed intermediate representation (Pydantic models +
   published JSON Schema), a validator, and a pluggable backend interface.
3. **Two reference backends** that prove portability:
   - **LangGraph** (reference target — its persistence primitives map 1:1 onto adhesives)
   - **Claude Agent SDK** (the batteries-included target — teams map onto subagents,
     budgets and isolation map onto hooks)

Plus a **conformance test suite**: golden `.glue` files with asserted behaviors
(e.g., "a `tape` result must not appear in any later context window"; "no message can
travel between repelled teams") that every backend must pass. This is what keeps
"compiles to multiple runtimes" an honest claim instead of marketing.

## 2. What the language looks like

The v1 surface survives almost intact; the changes fix semantic holes found in the audit
(nondeterministic adhesive selection, `pull` parsed as a team name, adhesives attached to
models instead of tool bindings):

```glue
app research_pipeline {
    version = "1.0"
    budget { max_usd = 5.00, max_turns = 40 }
}

// Tools are MCP servers or built-ins — declared, never implemented here.
mcp tavily {
    transport = stdio
    command   = "npx -y tavily-mcp"
    env       = [TAVILY_API_KEY]          // named, never inlined
}

tool web_search  { from = tavily }
tool file_writer { from = builtin.fs, root = "./out" }

model fast { provider = openrouter, id = "deepseek/deepseek-v3.2" }
model deep { provider = anthropic,  id = "claude-sonnet-5" }

team research {
    lead    = deep { role = "Plan the research; verify member findings." }
    members = [ fast * 3 { role = "Execute one search subtask." } ]
    tools   = [ web_search: velcro ]   // adhesive is per tool BINDING now
    publishes = report: glue           // typed artifact, team-persistent
}

team docs {
    lead  = deep { role = "Write the final brief from research artifacts." }
    tools = [ file_writer: tape ]      // one-shot; nothing retained
}

flow {
    research -> docs      // push published artifacts on completion
    docs     <- research  // docs may QUERY research's glue store on demand
    docs     <> finance   // repel: no channel exists, structurally
}
```

Key language changes from v1:

| v1 problem (from the audit) | v2 fix |
|---|---|
| Adhesives are a *set on the model*; runtime picks one nondeterministically (`next(iter(set))`) | Adhesive is declared **per tool binding** and **per published artifact** — unambiguous at compile time |
| `docs <- pull` parses `pull` as a target team name | `<-` takes a real team; pull semantics defined (see §3) |
| Flows outside `flow {}` crash the parser (the README syntax!) | Both forms parse; one canonical form emitted by `stickyc fmt` |
| VELCRO/TAPE are log statements | Each adhesive has a conformance-tested contract |
| No budgets, no isolation guarantees | `budget {}` blocks and `<>` compile to enforced hooks/policies |
| Bespoke fake-MCP layer | `mcp {}` blocks declare real Model Context Protocol servers; the backend wires them |

## 3. The semantic kernel (what backends must implement)

This table **is** the product. Everything else is syntax sugar around it.

| Concept | Guarantee | LangGraph mapping | Claude Agent SDK mapping |
|---|---|---|---|
| `glue` | Result persists team-wide across sessions; readable by all team members; survives process restart | `BaseStore` namespace `(team, artifact)` | Memory-tool directory scoped to the team |
| `velcro` | Result persists for the session/thread only; private to the binding agent | Thread checkpointer state | Session context (CLAUDE.md-adjacent scratch, cleared per session) |
| `tape` | Result is injected into exactly one turn, then unrecoverable | Tool result excluded from checkpoint | Tool result not written to any store |
| `A -> B` | A's published artifacts are delivered to B on completion | Graph edge carrying typed payload | Parent orchestration passes artifact to subagent |
| `B <- A` | B gets a **generated read-only tool** (`query_A_store`) over A's glue store; A is never interrupted | Auto-generated store-reader tool bound to B's lead | Same, as an MCP tool |
| `A <> B` | No channel exists; cross-team access is denied by machinery | No edge + store-access policy | `PreToolUse` hook denies by team label |
| `budget` | Hard ceiling on spend/turns; breach halts the run with a typed error | Middleware counting usage | Hooks + max-turns config |

The pull-becomes-a-generated-tool rule is worth highlighting: in v1, "pull" was a word in
a README. In v2 it is a mechanically derived capability — `docs <- research` *causes a
tool to exist*. That is the difference between documentation and compilation.

## 4. How it's delivered and how people use it

**Delivery**: `pip install stickyscript` (or `pipx`). One repo: spec, compiler,
backends, conformance suite, examples gallery. The IR's JSON Schema is published so
third parties can emit or consume compiled form without touching the DSL.

**Day-to-day workflow**:

```
stickyc init                      # scaffold app.glue
$EDITOR app.glue                  # declare teams, tools, flows, budgets
stickyc check app.glue            # static validation — the step markdown can't have
stickyc graph app.glue            # render the team topology as Mermaid/SVG
stickyc build app.glue --target langgraph   # emit a runnable Python project
stickyc build app.glue --target claude-agent-sdk  # emit agents/, settings, runner
stickyc run app.glue --input "..."          # batteries-included: build + execute
```

`stickyc check` catches, at compile time: references to undeclared models/tools/teams,
flows to nonexistent teams, cyclic pull chains, a `tape` artifact referenced by a
downstream team (impossible by definition), missing env declarations for MCP servers,
budget-less apps if a policy requires them. In CI, `stickyc check` runs on every PR —
the `.glue` file is the reviewed, diffed, versioned source of truth for the agent
org chart, and `stickyc graph` output drops into the PR description.

**Who adopts it**: platform teams standardizing how their org builds multi-agent
systems (one reviewable format instead of N bespoke Python files); agencies shipping
reproducible agent configs to clients; anyone who wants runtime portability (the same
file targets LangGraph today, Agent SDK tomorrow) without a rewrite.

## 5. Why this is not a fancy RULES.md

The comparison is the right one to press on, because a RULES.md / AGENTS.md and a
`.glue` file look superficially similar: both are checked-in text describing how agents
should behave. The difference is *who enforces it*:

- **A RULES.md is advisory prose consumed by a model at runtime.** Its only enforcement
  mechanism is model compliance. It cannot fail fast, cannot be validated, cannot be
  diffed semantically, and silently degrades — the model ignores a rule and nothing in
  the system knows.
- **A `.glue` file is compiled.** Five properties follow that prose cannot have:
  1. **It can fail before anything runs.** An undeclared tool or an impossible flow is a
     compile error, not a 3 a.m. runtime surprise.
  2. **Its guarantees are structural, not behavioral.** `docs <> finance` doesn't *ask*
     agents not to talk — the channel between them *does not exist*, and a hook denies
     any attempt at cross-team store access. Misbehavior is impossible, not discouraged.
  3. **Scoping is wiring, not intention.** Adhesives compile to which store a result is
     written to and which context windows it may enter. A model cannot "forget" the
     policy because the policy is not in its prompt — it's in the plumbing around it.
  4. **Budgets are enforced by middleware counting real tokens/dollars**, not by a
     sentence asking the model to be frugal.
  5. **It has a toolchain lifecycle**: format, lint, graph, diff, conformance-test,
     multi-target build. `stickyc graph` can render the org chart *because the file has
     structure*; nothing can render a RULES.md.

The honest boundary: role descriptions inside a `.glue` file are still prose, and still
advisory — StickyScript adds nothing there. Its value is the **structural layer**:
topology, memory scoping, tool provisioning, isolation, budgets. A single agent with a
system prompt doesn't need it; the moment there are multiple agents, shared state, and
boundaries that matter (data isolation, spend), prose stops being an enforcement
mechanism and compilation starts earning its keep.

One-line version: **RULES.md configures the model's intentions; StickyScript configures
the system the model runs inside.**

## 6. Build plan and honest risks

| Phase | Scope | Effort (single experienced dev) |
|---|---|---|
| 0 | Spec + parser salvage/fixes + IR + `check`/`fmt`/`graph` | 2–3 weeks |
| 1 | LangGraph backend + `build`/`run` + examples | 3–4 weeks |
| 2 | Claude Agent SDK backend + conformance suite | 2–3 weeks |
| 3 | LSP/VS Code highlighting, preset registry (common MCP servers), importers | ongoing |

Risks, stated plainly:
- **DSL adoption is hard.** Mitigation: the JSON IR is a first-class citizen — teams
  allergic to new syntax can write YAML/JSON against the schema and still get the
  validator, graph, and backends. The DSL is the ergonomic layer, not a hostage-taker.
- **The kernel is small enough to be absorbed** by LangGraph/CrewAI themselves. That is
  a success condition, not a failure: if "glue/velcro/tape" becomes the vocabulary other
  frameworks use for memory scoping, the idea won — which is precisely the part of
  GLUE-fw worth not abandoning.
- **Semantics drift across backends** is the credibility killer; the conformance suite
  exists to prevent it and must ship with Phase 2, not later.
