# Agno + GLUE

Question: We are building a GLUE multi agentic AI developer framework, and there is alot we still need to do a lot of basic core development for features that aren’t really unique are pretty standard for any current similar framework, but there is a Mozilla 2 licensed framework that is like, quite startlingly, so very aligned and similar to GLUE, it’s almost like if GLUE was just created to be another framework and wasn’t focused on the deeper future problems we see slowly showing up in the AI developer community, we have a deeper reason for our framework that is focused on preventing a huge conflict of interest in Agentic development and may lead to innovative stagnation and corporate enterprise monopoly like grips on the entire small time and startup community from being able to actually truly innovate and design increasingly modern solutions and innovations.

In fact this other framework just seems TOO non unique as in it doesn’t seem to offer anything that is new or attractive to use over say crew.ai or langchain…

What is your opinion on instead of building basically everything that we still need to (the core framework agnostic features of nearly all multi agentic developer frameworks) and patching our framework with Agno, and of course honoring their license, but it would literally free us up to have a fully functional and stable core framework where we could focus ONLY on the GLUE specific features that we believe will gain a lot of intention as the gap between proprietary Agentic agents becoming very advanced and vibe coding losing its edge because of a lack of understanding the code you’re building That is currently part of the process of vibe coding.

As agents get super advanced, we believe that a new type of standardized way to orchestrate advanced coding agents, by a new approach to prompt engineering and a DSL written on top of our framework that streamlines development by elegantly abstracting boilerplate into either implicit automations based on the near 99% of all AI Agent development use cases in Python building the basic configuration of their applications being the same. And dropping explicit API fetching from the .env file (which it still does, it just doesn’t need to be declared in the code, you set the model provider and the dsl parses it out to automatically check for a .env file and for any secret that starts with a capitalized version of the provider name.

It’s things and other things that leads me to the question I have for you, because if the seeming similarities and compatibility, I’m tempted to clone agno, and patch it into to the sections of GLUE that are not specific or unique to GLUE,

Therefore leaving us with a stable and working framework on which, since we are only two developers, we can focus solely on GLUE and the reasonings why we are creating GLUE, saving hours and hours of creating a stable working framework from scratch from which to build GLUE on top of.

Of course we would correctly include the open source license and all that,

What do you think of this idea in terms of a small startup like ours?

## Objective

Integrate the Agno framework as the core engine for the GLUE framework. The goal is to replace GLUE's existing core functionalities with Agno's robust and well-tested components (Agent, Team, Workflow) while preserving and building upon GLUE's unique features: the StickyScript DSL, the GLUE Forge CLI, Team Structures (with tool assignment at the team level), Adhesives (for memory/output management), and Magnetic Flow Operators (for inter-team communication).

This strategic move leverages Agno's architecture to streamline development, allowing the team to focus on innovating within the agentic AI space by enhancing GLUE's specific value propositions rather than reinventing core agentic functionalities.

Key Features to Retain/Build on Agno:

- **Magnetic Flow Operators**: For complex team-to-team communication patterns (PUSH, PULL, BIDIRECTIONAL, REPEL).

## Integration Approach

The integration will involve replacing `glue.core` components with their Agno counterparts (`agno.agent`, `agno.team`, `agno.workflow`). The existing GLUE features (DSL, CLI, Adhesives, Magnetic Flow) will be refactored to interface with the Agno core.

### Core Replacements

- **Adhesives**: Rework `glue.core.adhesive` to utilize Agno's memory or state management systems, potentially via agent context or hooks.
- **Magnetic Flow**: Adapt `glue.magnetic.field` to orchestrate communication between Agno `Team` instances, possibly requiring extensions or wrappers around Agno `Team`.

### Next Steps

- **DSL Integration**: Integrate StickyScript DSL with Agno's workflow management to enable seamless interaction between GLUE's DSL and Agno's workflow engine.
- **CLI Refactoring**: Refactor the GLUE Forge CLI to work with Agno's agent and team management, ensuring a cohesive developer experience.

## PerplexityAI answer

Integrating Agno into your GLUE framework could be a strategically sound decision for your startup, given the constraints and goals described. Here's a structured analysis:

### Key Advantages of Adopting Agno

1. **Accelerated Development Timeline**
    - Agno provides battle-tested core features like multi-agent coordination, memory management, and model/provider agnosticism out of the box [2, 10, 16].
    - By leveraging its existing infrastructure, you avoid reinventing foundational components (e.g., tool integration, monitoring systems) [14], freeing 6-12 months of development time for two developers.
2. **Performance Benefits**
    - Agno's 5,000x faster agent instantiation and 50x lower memory usage compared to LangGraph [2, 16] give GLUE a technical edge from day one – critical for startups competing with larger players.
3. **License Compatibility**
    - The Mozilla Public License 2.0 permits modification and redistribution [9, 12], aligning with your goal to build an open ecosystem while allowing commercial use cases.
4. **Focus on Differentiation**
    - You could redirect efforts to GLUE's unique value:
        - **DSL for Agent Orchestration** (your proposed abstraction layer)
        - **Ethical Guardrails & Provenance** (your unique market positioning)
        - **Vibe Coding Mitigation** through new prompt engineering technique

### Strategic Considerations

| Factor | Agno Integration | Custom Build |
| --- | --- | --- |
| **Time-to-Market** | Faster (6-12 months saved) | Slower (requires core feature build) |
| **Resource Needs** | Lower (leverage existing codebase) | Higher (build everything from scratch) |
| **Community Leverage** | Access Agno's 18.5k+ GitHub community [2] | Must build ecosystem from scratch |
| **Innovation Focus** | 80% effort on GLUE-specific features | ≤50% effort on commodity features |

### Implementation Recommendations

1. **Architecture Strategy**
    - Fork Agno's core and rebrand as GLUE Core [12], maintaining compliance with MPL-2.0 attribution requirements.
    - Build your DSL and ethical orchestration layer as modular extensions [16], ensuring clean separation from upstream components.
2. **Risk Mitigation**
    - Create abstraction layers between Agno and GLUE-specific code to limit vendor lock-in [8].
    - Contribute critical improvements back to Agno's main repo to reduce fork maintenance [11].
3. **Go-to-Market Edge**
    - Position GLUE as "Agno++" – leveraging its performance while adding:
        - New type of prompt engineering to orchestrate agents with better human codebase understanding to combat vibe coding
        - Ethical guardrails & transparency features (missing in Agno/CrewAI/LangChain [5, 10])
        - Visual workflow builder for non-technical users [10, 15]

### Conclusion

For a resource-constrained startup, integrating Agno is a pragmatic choice that balances technical ambition with execution feasibility. By treating Agno as your foundation rather than competition, you create space to solve higher-order problems in AI ethics and developer ergonomics – areas where existing frameworks like LangChain/CrewAI show limited vision [5, 10]. Just ensure your licensing documentation clearly communicates the relationship between GLUE and Agno to maintain community trust [6, 11].

Sources
[1] mozilla-ai/any-agent: A single interface to build and ... - GitHub [https://github.com/mozilla-ai/any-agent](https://github.com/mozilla-ai/any-agent)
[2] Building AI Agents with Agno-Phidata (Github +18.5k Stars) [https://dev.to/mehmetakar/building-ai-agents-with-agno-phidata-tutorial-4ilh](https://dev.to/mehmetakar/building-ai-agents-with-agno-phidata-tutorial-4ilh)
[3] Agno: EASILY Build Agents with Memory, Knowledge ... - YouTube [https://www.youtube.com/watch?v=XN6dSSx6Ehg](https://www.youtube.com/watch?v=XN6dSSx6Ehg)
[4] Top 12 Frameworks for Building AI Agents in 2025 - Bright Data [https://brightdata.com/blog/ai/best-ai-agent-frameworks](https://brightdata.com/blog/ai/best-ai-agent-frameworks)
[5] Langchain vs CrewAI: Comparative Framework Analysis - [Orq.ai](http://orq.ai/) [https://orq.ai/blog/langchain-vs-crewai](https://orq.ai/blog/langchain-vs-crewai)
[6] 7 Best Practices to Secure Your Open Source Components [https://www.salesforce.com/blog/secure-open-source/](https://www.salesforce.com/blog/secure-open-source/)
[7] Build or Use Open-Source Software: An Expert Guide - Finoit [https://www.finoit.com/articles/build-vs-open-source-software/](https://www.finoit.com/articles/build-vs-open-source-software/)
[8] AI for Startups - Microsoft On the Issues [https://blogs.microsoft.com/on-the-issues/2024/11/01/ai-for-startups/](https://blogs.microsoft.com/on-the-issues/2024/11/01/ai-for-startups/)
[9] Mozilla AI just released a Python library to access multiple AI Agent ... [https://www.threads.net/@_saboo_shubham/post/DInIcq2sLpe/mozilla-ai-just-released-a-python-library-to-access-multiple-ai-agent-frameworks](https://www.threads.net/@_saboo_shubham/post/DInIcq2sLpe/mozilla-ai-just-released-a-python-library-to-access-multiple-ai-agent-frameworks)
[10] 7 Most Trending Frameworks for Building Multi-Agent AI Applications [https://aiagent.marktechpost.com/post/7-best-frameworks-for-building-multi-agent-ai-apps](https://aiagent.marktechpost.com/post/7-best-frameworks-for-building-multi-agent-ai-apps)
[11] What are some best practices for open source developers? - MoldStud [https://moldstud.com/articles/p-what-are-some-best-practices-for-open-source-developers](https://moldstud.com/articles/p-what-are-some-best-practices-for-open-source-developers)
[12] All repositories - agno-agi - GitHub [https://github.com/orgs/agno-agi/repositories](https://github.com/orgs/agno-agi/repositories)
[13] Open-Source Strategies: A 2025 Guide for Small Business Success [https://www.linkedin.com/pulse/open-source-strategies-2025-guide-small-business-success-leqlf](https://www.linkedin.com/pulse/open-source-strategies-2025-guide-small-business-success-leqlf)
[14] agno · PyPI [https://pypi.org/project/agno/](https://pypi.org/project/agno/)
[15] Best 5 Frameworks To Build Multi-Agent AI Applications - [GetStream.io](http://getstream.io/) [https://getstream.io/blog/multiagent-ai-frameworks/](https://getstream.io/blog/multiagent-ai-frameworks/)
[16] Unleashing the Power of Agno: Building Multi-Modal Agents with a ... [https://rpabotsworld.com/agno-building-multi-modal-agents-with-a-lightweight-python-library/](https://rpabotsworld.com/agno-building-multi-modal-agents-with-a-lightweight-python-library/)
[17] Agno is a lightweight library for building Agents with memory ... [https://github.com/agno-agi/agno](https://github.com/agno-agi/agno)
[18] Agno (Phidata) - Framework AI Agent Builder [https://bestaiagents.ai/agent/agno](https://bestaiagents.ai/agent/agno)
[19] 14 AI projects to watch: Mozilla's first Builders Accelerator cohort ... [https://blog.mozilla.org/en/mozilla/14-ai-projects-to-watch-mozillas-first-builders-accelerator-cohort-kicks-off/](https://blog.mozilla.org/en/mozilla/14-ai-projects-to-watch-mozillas-first-builders-accelerator-cohort-kicks-off/)
[20] Agno - AI Agent Reviews, Features, Use Cases & Alternatives (2025) [https://aiagentsdirectory.com/agent/agno](https://aiagentsdirectory.com/agent/agno)
[21] Agno - GitHub [https://github.com/agno-agi](https://github.com/agno-agi)
[22] Programs - Mozilla Builders [https://builders.mozilla.org/programs/](https://builders.mozilla.org/programs/)
[23] Agentic AI Tutorial One shot Using Agno(Phidata) - YouTube [https://www.youtube.com/watch?v=Kk4ukdWqKPM](https://www.youtube.com/watch?v=Kk4ukdWqKPM)
[24] Top 5 Open Source Frameworks for building AI Agents - Athina AI Hub [https://hub.athina.ai/top-5-open-source-frameworks-for-building-ai-agents-with-examples/](https://hub.athina.ai/top-5-open-source-frameworks-for-building-ai-agents-with-examples/)
[25] Best Agent Framework for Complex Agentic RAG Implementation [https://www.reddit.com/r/AI_Agents/comments/1jjqfgq/best_agent_framework_for_complex_agentic_rag/](https://www.reddit.com/r/AI_Agents/comments/1jjqfgq/best_agent_framework_for_complex_agentic_rag/)
[26] Top 5 Open Source Frameworks for building AI Agents - Reddit [https://www.reddit.com/r/AI_Agents/comments/1ip4uii/top_5_open_source_frameworks_for_building_ai/](https://www.reddit.com/r/AI_Agents/comments/1ip4uii/top_5_open_source_frameworks_for_building_ai/)
[27] Agno (@AgnoAgi) / X [https://x.com/agnoagi](https://x.com/agnoagi)
[28] Mastering Agents: LangGraph Vs Autogen Vs Crew AI - Galileo AI [https://www.galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew](https://www.galileo.ai/blog/mastering-agents-langgraph-vs-autogen-vs-crew)
[29] Best AI Agent Frameworks in 2025: A Comprehensive Guide - Reddit [https://www.reddit.com/r/AI_Agents/comments/1hq9il6/best_ai_agent_frameworks_in_2025_a_comprehensive/](https://www.reddit.com/r/AI_Agents/comments/1hq9il6/best_ai_agent_frameworks_in_2025_a_comprehensive/)
[30] dylanhogg/awesome-python - GitHub [https://github.com/dylanhogg/awesome-python](https://github.com/dylanhogg/awesome-python)
[31] [Crew.ai](http://crew.ai/) vs Langchain: A Thorough Look at Multi-Agent Frameworks [https://www.flowhunt.io/blog/crew-ai-vs-langchain-a-thorough-look-at-multi-agent-frameworks/](https://www.flowhunt.io/blog/crew-ai-vs-langchain-a-thorough-look-at-multi-agent-frameworks/)
[32] Top 9 AI Agent Frameworks as of April 2025 - Shakudo [https://www.shakudo.io/blog/top-9-ai-agent-frameworks](https://www.shakudo.io/blog/top-9-ai-agent-frameworks)
[33] 12-factor Agents: Patterns of reliable LLM applications | Hacker News [https://news.ycombinator.com/item?id=43699271](https://news.ycombinator.com/item?id=43699271)
[34] Langgraph vs CrewAI vs AutoGen vs PydanticAI vs Agno vs OpenAI ... [https://www.reddit.com/r/LangChain/comments/1jpk1vn/langgraph_vs_crewai_vs_autogen_vs_pydanticai_vs/](https://www.reddit.com/r/LangChain/comments/1jpk1vn/langgraph_vs_crewai_vs_autogen_vs_pydanticai_vs/)
[35] 8 Best Multi-Agent AI Frameworks for 2025 - Multimodal.dev [https://www.multimodal.dev/post/best-multi-agent-ai-frameworks](https://www.multimodal.dev/post/best-multi-agent-ai-frameworks)
[36] AgentFlow vs Crew AI vs Autogen vs LangChain for Building AI Agents [https://www.ankursnewsletter.com/p/agentflow-vs-crew-ai-vs-autogen-vs](https://www.ankursnewsletter.com/p/agentflow-vs-crew-ai-vs-autogen-vs)
[37] Top AI Agent frameworks and platforms in 2025 - WorkOS [https://workos.com/blog/top-ai-agent-frameworks-and-platforms-in-2025](https://workos.com/blog/top-ai-agent-frameworks-and-platforms-in-2025)
[38] Comparing Open-Source AI Agent Frameworks - Langfuse Blog [https://langfuse.com/blog/2025-03-19-ai-agent-comparison](https://langfuse.com/blog/2025-03-19-ai-agent-comparison)
[39] LLM Agent Frameworks 2025: Guide & Comparison - Chatbase [https://www.chatbase.co/blog/llm-agent-framework-guide](https://www.chatbase.co/blog/llm-agent-framework-guide)
[40] The similarities between open source work and running a tech startup. [https://tannerlinsley.com/posts/the-similarities-between-open-source-work-and-running-a-tech-startup](https://tannerlinsley.com/posts/the-similarities-between-open-source-work-and-running-a-tech-startup)
[41] Benefits and Applications of Open Source AI | Moesif Blog [https://www.moesif.com/blog/technical/api-development/Open-Source-AI/](https://www.moesif.com/blog/technical/api-development/Open-Source-AI/)
[42] How to safely open-source internal software - Some best practices [https://www.reddit.com/r/programming/comments/mzj57l/how_to_safely_opensource_internal_software_some/](https://www.reddit.com/r/programming/comments/mzj57l/how_to_safely_opensource_internal_software_some/)
[43] Grow Startup with Open Source Software: 6 Key Advantages [http://www.webdatarocks.com/blog/open-source-software-for-startups/](http://www.webdatarocks.com/blog/open-source-software-for-startups/)
[44] Open source technology in the age of AI - McKinsey & Company [https://www.mckinsey.com/capabilities/quantumblack/our-insights/open-source-technology-in-the-age-of-ai](https://www.mckinsey.com/capabilities/quantumblack/our-insights/open-source-technology-in-the-age-of-ai)
[45] 10up Open Source Best Practices [https://10up.github.io/Open-Source-Best-Practices/](https://10up.github.io/Open-Source-Best-Practices/)
[46] Why every startup should be built on open-source... - DEV Community [https://dev.to/github20k/why-every-startup-should-be-built-on-open-source-2kh](https://dev.to/github20k/why-every-startup-should-be-built-on-open-source-2kh)
[47] The Rise Of Open Artificial Intelligence: Open-Source Best Practices [https://www.forbes.com/councils/forbesbusinesscouncil/2024/03/08/the-rise-of-open-artificial-intelligence-open-source-best-practices/](https://www.forbes.com/councils/forbesbusinesscouncil/2024/03/08/the-rise-of-open-artificial-intelligence-open-source-best-practices/)
[48] Best practices for contributing to Open Source? : r/learnprogramming [https://www.reddit.com/r/learnprogramming/comments/t4dm7h/best_practices_for_contributing_to_open_source/](https://www.reddit.com/r/learnprogramming/comments/t4dm7h/best_practices_for_contributing_to_open_source/)
[49] [Advice] How do you create a business out of an open source ... [https://www.reddit.com/r/opensource/comments/1e0uy2p/advice_how_do_you_create_a_business_out_of_an/](https://www.reddit.com/r/opensource/comments/1e0uy2p/advice_how_do_you_create_a_business_out_of_an/)
[50] Practical Considerations in Choosing Open-Source or Closed ... [https://www.wilsonelser.com/publications/practical-considerations-in-choosing-open-source-or-closed-source-ai-for-business-workflows](https://www.wilsonelser.com/publications/practical-considerations-in-choosing-open-source-or-closed-source-ai-for-business-workflows)
[51] The Benefits of Open Source Software for Small Businesses - Idalko [https://idalko.com/open-source-software-benefits/](https://idalko.com/open-source-software-benefits/)
[52] Why are developers moving away from LangChain? - Reddit [https://www.reddit.com/r/LangChain/comments/1j1gb88/why_are_developers_moving_away_from_langchain/](https://www.reddit.com/r/LangChain/comments/1j1gb88/why_are_developers_moving_away_from_langchain/)
[53] Why we no longer use LangChain for building our AI agents [https://news.ycombinator.com/item?id=40739982](https://news.ycombinator.com/item?id=40739982)
[54] AI Agent Frameworks to Watch in 2025 - Curotec [https://www.curotec.com/insights/ai-agent-frameworks/](https://www.curotec.com/insights/ai-agent-frameworks/)
[55] The Multi-Agent Revolution: 5 AI Frameworks That Are ... - Fluid AI [https://www.fluid.ai/blog/the-multi-agent-revolution-5-ai-frameworks](https://www.fluid.ai/blog/the-multi-agent-revolution-5-ai-frameworks)
[56] Comparing Top Open-Source AI Agent Frameworks & Platforms [https://www.helicone.ai/blog/ai-agent-builders](https://www.helicone.ai/blog/ai-agent-builders)
[57] Strengthening Open Source Software: Best Practices for Enhanced ... [https://openssf.org/blog/2023/09/06/strengthening-open-source-software-best-practices-for-enhanced-security/](https://openssf.org/blog/2023/09/06/strengthening-open-source-software-best-practices-for-enhanced-security/)
[58] Best Practices for Setup and Management of an Open Source Project [https://stackoverflow.com/questions/2567766/best-practices-for-setup-and-management-of-an-open-source-project](https://stackoverflow.com/questions/2567766/best-practices-for-setup-and-management-of-an-open-source-project)
[59] Setting an Open Source Strategy - Linux Foundation [https://www.linuxfoundation.org/resources/open-source-guides/setting-an-open-source-strategy](https://www.linuxfoundation.org/resources/open-source-guides/setting-an-open-source-strategy)