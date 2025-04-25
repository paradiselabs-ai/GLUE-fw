# GLUE Framework: Feature Focus & Prioritization Guide

## Purpose
This guide is for all GLUE developers and contributors. It provides a clear, honest assessment of where to focus our energy for maximum impact, what to improve, and what to consider cutting or de-emphasizing. The goal: make GLUE a highly valuable, widely used tool by delivering on what matters most to users and the market.

---

## 1. Double Down: Core Differentiators & Growth Drivers

| Area                        | Why Focus Here?                                                                                         | Immediate Actions                                               |
|-----------------------------|--------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| Team/meta-team architecture | Unique to GLUE; enables scalable, organized AI.                                                        | Polish, document, build demos showing real-world teamwork.      |
| Dynamic tool/MCP creation   | Rare in frameworks; critical for enterprise/power users.                                                | Refine UX, add templates, showcase use cases in docs/demos.     |
| StickyScript DSL            | Lowers barrier, accelerates dev, appeals to broad skill levels.                                         | Improve docs, error messages, add lots of examples.             |
| Onboarding & documentation  | Key to adoption—users must see value quickly.                                                           | Create quick-starts, video walkthroughs, CLI onboarding.        |
| Performance/scaling         | Needed for enterprise; proves GLUE can handle real workloads.                                           | Benchmark, optimize, publish results.                           |
| Community/ecosystem         | Drives adoption, support, and innovation.                                                               | Lower contribution barriers, highlight community work.          |

---

## 2. Improve or Automate: Remove Friction

| Area                     | Why Improve?                              | Immediate Actions                       |
|--------------------------|-------------------------------------------|-----------------------------------------|
| Progress/test tracking   | Manual updates are error-prone, unscalable| Script or automate Pass List/progress.  |
| Provider integration     | Too many too soon = support burden        | Focus on most-used, expand as needed.   |

---

## 3. Pull Back or Reconsider: Avoid Bloat & Distraction

| Area                        | Why Pull Back?                                            | Action                                     |
|-----------------------------|----------------------------------------------------------|---------------------------------------------|
| Over-complex/niche features | Maintenance/onboarding cost, little user value           | Deprecate or move to contrib/experimental.  |
| Premature optimization      | Wastes time/resources; real needs come from users        | Ship MVPs, iterate based on feedback.       |

---

## 4. How to Decide: Keep, Improve, or Cut

- **Keep:** If it’s core to team-based, scalable, extensible agentic AI, or drives real user value.
- **Improve:** If it’s a friction point for onboarding, usability, or maintainability.
- **Cut:** If it’s little-used, confusing, or not aligned with GLUE’s vision.

**Always listen to real users and contributors. Simplicity and polish matter more than feature count.**

---

## 5. Next Steps
- Review all features/systems against this guide.
- Prioritize roadmap items that double down on our differentiators.
- Automate or simplify wherever possible.
- Be open to deprecating or moving non-core features.
- Foster a welcoming, active community.

---

*This guide should be revisited regularly as GLUE and its user base grow. For questions or to propose changes, open an issue or discuss in the team channel.*
