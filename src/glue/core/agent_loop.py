"""
Agent Loop module for the GLUE framework.

"""

from typing import List, Dict, Any, Callable, Tuple, Optional
import json
from .working_memory import WorkingMemory
from .schemas import ParseAnalyzeOutput, PlanPhaseOutput, ToolSelectionOutput, MemoryDecisionOutput, SelfEvalOutput, FormatResultOutput
from ..utils.json_utils import extract_json

class TeamMemberAgentLoop:
    """loop for Team Member agents. Tools given are report_task_completion tool."""
    def __init__(self, agent_id: str, team_id: str, report_tool: callable, agent_llm: Optional[Any] = None):
        self.agent_id = agent_id
        self.team_id = team_id
        self.report_tool = report_tool
        # LLM interface for analysis and planning
        self.agent_llm = agent_llm
        # Registry for external tools
        self.tools: Dict[str, Callable] = {}
        # Loop termination state
        self.terminated: bool = False
        self.termination_reason: Optional[str] = None
        # Tracking for refine/revise logic
        self.last_substep_description: Optional[str] = None
        self.last_context_info: Optional[Dict[str, Any]] = None
        self.last_self_eval: Optional[SelfEvalOutput] = None
        self.last_tool_name: Optional[str] = None
        self.last_tool_params: Optional[Dict[str, Any]] = None
        self.state = {'current_task_id': None, 'status': 'idle'}
        # Initialize working memory and control counters
        self.working_memory = WorkingMemory()
        self.turn_counter: int = 0
        self.max_attempts: int = 3  # cap on loop iterations
        self.substeps: List[str] = []  # ordered list of substeps from plan
        self.current_step: int = 0  # index of the current substep being executed
        self.attempts_per_step: Dict[int, int] = {}  # track number of attempts per substep

    async def start(self, fetch_task_func: callable):
        """Start the Team Member loop to perform tasks and report completion."""
        # Fetch the assigned task once
        task = await fetch_task_func(self.agent_id)
        self.state['current_task_id'] = task['task_id']
        self.state['status'] = 'working'
        # Gather initial context
        self.gather_context(task)
        # Parse and analyze requirements
        analysis = await self.parse_and_analyze(task['description'])
        self.working_memory.add_entry(self.turn_counter, str(analysis), 'ParseAnalyze')
        # Plan execution stub and initialize substeps
        planned_steps = await self.plan_phase(self.working_memory.get_entries())
        self.working_memory.add_entry(self.turn_counter, str(planned_steps), 'PlanPhase')
        self.substeps = planned_steps.substeps
        self.current_step = 0
        self.attempts_per_step = {}
        while True:
            # Check for external termination
            if self.terminated:
                break
            # Increment turn counter for substep execution
            self.turn_counter += 1
            # AccessMemory stage: prepare context entries
            memory_entries = self.working_memory.get_entries()
            raw_entries = [e for e in memory_entries if e['source_tool'] != 'Curated']
            recent_raw = raw_entries[-self.max_attempts:] if raw_entries else []
            context_info = {
                'turn': self.turn_counter,
                'memory_entries': memory_entries,
                'recent_raw_outputs': recent_raw
            }
            # Execute selected tool for current substep with context
            substep_description = self.substeps[self.current_step]
            tool_name, result = await self.select_and_invoke_tool(substep_description, context_info)
            # Auto-save raw output
            self.working_memory.add_entry(self.turn_counter, str(result), tool_name)
            # Memory decision hook
            memory_decision = await self.should_save_to_memory(result)
            if memory_decision.save_to_memory:
                curated = memory_decision.analysis or ''
                self.working_memory.add_entry(self.turn_counter, curated, 'Curated')
            # Self-evaluate result and track attempts
            eval_out = await self.evaluate_self(result)
            # Store full self-eval output
            self.last_self_eval = eval_out
            confidence = eval_out.confidence_level.lower()
            self.attempts_per_step[self.current_step] = self.attempts_per_step.get(self.current_step, 0) + 1
            if confidence == 'high':
                # High confidence: proceed to next substep or finish
                if self.current_step < len(self.substeps) - 1:
                    self.current_step += 1
                    continue
                # Last substep: format and report success
                detailed_answer = await self.format_result(result)
                final_status = 'success' if result.get('success') else 'failure'
                self.state['status'] = 'reporting'
                await self.report_tool(
                    task_id=self.state['current_task_id'],
                    status=final_status,
                    detailed_answer=detailed_answer,
                    artifact_keys=result.get('artifacts', []),
                    calling_agent_id=self.agent_id,
                    calling_team=self.team_id
                )
                self.state['status'] = 'completed'
                break
            elif confidence == 'medium':
                # Medium confidence: refine and retry
                result = await self.refine_result(result)
                continue
            elif confidence == 'low':
                # Low confidence: revise strategy or report partial failure
                if self.attempts_per_step[self.current_step] < self.max_attempts:
                    result = await self.revise_strategy(result)
                    continue
                # Exceeded max attempts: report failure with limitations
                detailed_answer = await self.format_result(result)
                self.state['status'] = 'reporting'
                await self.report_tool(
                    task_id=self.state['current_task_id'],
                    status='failure',
                    detailed_answer=detailed_answer,
                    artifact_keys=result.get('artifacts', []),
                    calling_agent_id=self.agent_id,
                    calling_team=self.team_id
                )
                self.state['status'] = 'completed'
                break
            else:
                # Critical error: escalate and report failure
                detailed_answer = await self.format_result(result)
                self.state['status'] = 'reporting'
                await self.report_tool(
                    task_id=self.state['current_task_id'],
                    status='failure',
                    detailed_answer=detailed_answer,
                    artifact_keys=[],
                    calling_agent_id=self.agent_id,
                    calling_team=self.team_id
                )
                self.state['status'] = 'completed'
                break

    async def performTaskStub(self, task_description: str) -> dict:
        """Placeholder for performing the task (simulated)."""
        import asyncio
        # Simulate work
        await asyncio.sleep(1)
        # Hardcoded success result with placeholder artifacts
        return {'success': True, 'artifacts': []}

    def generateSummaryStub(self, result: dict) -> str:
        """Placeholder for generating a detailed answer based on the result."""
        # Provide details including success status and any artifacts or errors
        if result.get('success'):
            artifacts = result.get('artifacts', [])
            return f"Task completed successfully with artifacts: {artifacts}" 
        else:
            # Include any error message if present
            error_msg = result.get('error', 'Unknown error')
            return f"Task failed: {error_msg}"

    def gather_context(self, task: Dict[str, Any]):
        """Collect initial context from the task description."""
        self.working_memory.add_entry(self.turn_counter, task['description'], 'TaskReceived')

    async def parse_and_analyze(self, description: str) -> ParseAnalyzeOutput:
        """Use LLM to extract high-level requirements from task description as structured JSON."""
        if not self.agent_llm:
            # Fallback stub without LLM
            keywords = description.split()[:3]
            return ParseAnalyzeOutput(
                thought_process="Fallback stub analysis",
                analysis={"keywords": keywords}
            )
        # Construct prompt requesting structured JSON output per schema
        prompt = (
            "Provide ONLY a fenced JSON response matching the ParseAnalyzeOutput schema,\n"
            "with fields:\n"
            "- thought_process (string)\n"
            "- analysis (object)\n"
            "Example:\n```json\n"
            "{\"thought_process\":\"parsed reasoning\",\"analysis\":{\"key\":\"value\"}}\n"
            "```\n"
            "Now analyze the following task description:\n"
            f"{description}\n"
        )
        # Invoke LLM generation
        response = await self.agent_llm.generate(prompt)
        # Attempt to extract JSON from response
        parsed = extract_json(response)
        if parsed:
            try:
                return ParseAnalyzeOutput.parse_obj(parsed)
            except Exception:
                pass
        # Fallback to direct JSON loads
        try:
            data = json.loads(response)
            return ParseAnalyzeOutput.parse_obj(data)
        except Exception:
            # Final stub fallback
            return ParseAnalyzeOutput(
                thought_process="Fallback stub analysis",
                analysis={"analysis": response}
            )

    async def plan_phase(self, memory: List[Dict[str, Any]]) -> PlanPhaseOutput:
        """Use LLM to break memory entries into substeps, tool requirements, and estimated confidence as structured JSON."""
        if not self.agent_llm:
            # Fallback stub without LLM
            return PlanPhaseOutput(
                substeps=["Step 1", "Step 2", "Step 3"],
                tool_requirements=[],
                estimated_confidence="high"
            )
        # Prompt LLM for structured JSON output per schema
        prompt = (
            "Provide ONLY a fenced JSON response matching the PlanPhaseOutput schema,\n"
            "with fields:\n"
            "- substeps (array of strings)\n"
            "- tool_requirements (array of strings)\n"
            "- estimated_confidence (string)\n"
            "Example:\n```json\n"
            "{\"substeps\":[\"Step 1\",\"Step 2\"],\"tool_requirements\":[\"tool1\"],\"estimated_confidence\":\"high\"}\n"
            "```\n"
            "Plan execution for the following memory entries:\n"
            f"{json.dumps(memory)}\n"
        )
        response = await self.agent_llm.generate(prompt)
        # Attempt to extract JSON from response
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            try:
                return PlanPhaseOutput.parse_obj(parsed)
            except Exception:
                pass
        # Fallback to direct JSON loads
        try:
            data = json.loads(response)
            return PlanPhaseOutput.parse_obj(data)
        except Exception:
            # Final stub fallback
            return PlanPhaseOutput(
                substeps=["Step 1", "Step 2", "Step 3"],
                tool_requirements=[],
                estimated_confidence="high"
            )

    async def select_and_invoke_tool(self, task_description: str, context_info: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
        """Use LLM to select the next tool and invoke it based on structured JSON output."""
        # Track substep and context
        self.last_substep_description = task_description
        self.last_context_info = context_info
        if not self.agent_llm:
            # Fallback stub execution
            self.last_tool_name = 'performTaskStub'
            self.last_tool_params = {'task_description': task_description}
            result = await self.performTaskStub(task_description)
            return 'performTaskStub', result
        # Prompt LLM for tool selection JSON, including memory context if available
        prompt = (
            "Provide ONLY a fenced JSON response matching the ToolSelectionOutput schema,\n"
            "with fields:\n"
            "- selected_tool_name (string)\n"
            "- tool_parameters (object)\n"
            "Example:\n```json\n"
            "{\"selected_tool_name\":\"performTask\",\"tool_parameters\":{\"param\":\"value\"}}\n"
            "```\n"
            "Choose the next tool for the following substep and context.\n"
            f"Substep: {task_description}\n"
        )
        if context_info is not None:
            prompt += f"Context: {json.dumps(context_info)}"
        response = await self.agent_llm.generate(prompt)
        parsed = extract_json(response)
        selection = None
        if isinstance(parsed, dict):
            try:
                selection = ToolSelectionOutput.parse_obj(parsed)
            except Exception:
                selection = None
        # Invoke selected tool or fallback
        if selection:
            tool_name = selection.selected_tool_name
            params = selection.tool_parameters or {}
            # Track tool invocation
            self.last_tool_name = tool_name
            self.last_tool_params = params
            # First try internal method
            if hasattr(self, tool_name) and callable(getattr(self, tool_name)):
                try:
                    result = await getattr(self, tool_name)(**params)
                    return tool_name, result
                except Exception:
                    pass
            # Then try external tool registry
            if tool_name in self.tools and callable(self.tools[tool_name]):
                try:
                    result = await self.tools[tool_name](**params)
                    return tool_name, result
                except Exception:
                    pass
        # Fallback to performTaskStub on failure
        result = await self.performTaskStub(task_description)
        return 'performTaskStub', result

    async def should_save_to_memory(self, result: Any) -> MemoryDecisionOutput:
        """Prompt agent to decide if result should be saved using the MemoryDecisionOutput schema."""
        if not self.agent_llm:
            return MemoryDecisionOutput(save_to_memory=False)
        prompt = (
            "Provide ONLY a fenced JSON response matching the MemoryDecisionOutput schema,\n"
            "with fields:\n"
            "- save_to_memory (boolean)\n"
            "- analysis (string)\n"
            "Example:\n```json\n"
            "{\"save_to_memory\":true,\"analysis\":\"key details\"}\n"
            "```\n"
            "Decide whether to save the following tool result:\n"
            f"{result}\n"
        )
        response = await self.agent_llm.generate(prompt)
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            try:
                return MemoryDecisionOutput.parse_obj(parsed)
            except Exception:
                pass
        # Fallback stub
        return MemoryDecisionOutput(save_to_memory=False)

    async def evaluate_self(self, result: Any) -> SelfEvalOutput:
        """Use LLM to self-evaluate and return a SelfEvalOutput object."""
        # Default stub
        if not self.agent_llm:
            stub = SelfEvalOutput(
                evaluation_summary="Fallback stub self-evaluation",
                consistency_check="Passed",
                alignment_check="Passed",
                confidence_level="HighConfidence",
                error_detected=False
            )
            return stub
        # Construct prompt for SelfEvalOutput
        prompt = (
            "Please output ONLY valid JSON matching the SelfEvalOutput schema "
            "(fields: evaluation_summary, consistency_check, alignment_check, confidence_level, error_detected) "
            "for the following result and context entries:\n"
            f"Result: {result}\n"
            f"Context: {json.dumps(self.working_memory.get_entries())}"
        )
        response = await self.agent_llm.generate(prompt)
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            try:
                return SelfEvalOutput.parse_obj(parsed)
            except Exception:
                pass
        # Fallback stub if parsing failed
        stub = SelfEvalOutput(
            evaluation_summary="Fallback stub self-evaluation",
            consistency_check="Passed",
            alignment_check="Passed",
            confidence_level="HighConfidence",
            error_detected=False
        )
        return stub

    async def refine_result(self, last_result: Any) -> Any:
        """Refine the last result by re-invoking the substep tool with feedback."""
        # Gather feedback from last self-evaluation
        feedback = self.last_self_eval.evaluation_summary if self.last_self_eval else ''
        # Augment context with refine feedback
        context = dict(self.last_context_info) if self.last_context_info else {}
        context['refine_feedback'] = feedback
        # Re-invoke tool for the same substep
        _, new_result = await self.select_and_invoke_tool(
            self.last_substep_description, context
        )
        return new_result

    async def revise_strategy(self, last_result: Any) -> Any:
        """Revise strategy for the current substep by planning a new approach."""
        # Gather failure reason from last self-evaluation
        reason = self.last_self_eval.evaluation_summary if self.last_self_eval else ''
        # Augment context with failure reason for replanning
        context = dict(self.last_context_info) if self.last_context_info else {}
        context['failure_reason'] = reason
        # Re-invoke planning/execution for the same substep
        _, new_result = await self.select_and_invoke_tool(
            self.last_substep_description, context
        )
        return new_result

    async def format_result(self, result: Any) -> str:
        """Use LLM to format the final result as JSON matching the FormatResultOutput schema."""
        entries = self.working_memory.get_entries()
        # Fallback stub
        if not self.agent_llm:
            fmt = FormatResultOutput(
                final_answer=str(result),
                supporting_context=[str(e) for e in entries]
            )
            return fmt.json()
        prompt = (
            "Provide ONLY a fenced JSON response matching the FormatResultOutput schema,\n"
            "with fields:\n"
            "- final_answer (string)\n"
            "- supporting_context (array of strings)\n"
            "Example:\n```json\n"
            "{\"final_answer\":\"...\",\"supporting_context\":[\"...\",\"...\"]}\n"
            "```\n"
            "Format the final result using the following data:\n"
            f"Result: {result}\n"
            f"Context entries: {json.dumps(entries)}\n"
        )
        response = await self.agent_llm.generate(prompt)
        parsed = extract_json(response)
        if isinstance(parsed, dict):
            try:
                fmt_out = FormatResultOutput.parse_obj(parsed)
                return fmt_out.json()
            except Exception:
                pass
        # Final stub fallback
        fmt = FormatResultOutput(
            final_answer=str(result),
            supporting_context=[str(e) for e in entries]
        )
        return fmt.json()

    def register_tool(self, name: str, func: Callable):
        """Register an external tool callable under the given name."""
        self.tools[name] = func

    def terminate(self, reason: Optional[str] = None):
        """Externally terminate the loop execution."""
        self.terminated = True
        self.termination_reason = reason

class TeamLeadAgentLoop:
    """loop for Team Lead agents. Tools given are delegate_task tool."""
    def __init__(self, agent_id: str, team_id: str, delegate_tool: callable, member_ids: list):
        self.agent_id = agent_id
        self.team_id = team_id
        self.delegate_tool = delegate_tool
        self.member_ids = member_ids
        self.state = {'parent_task_id': None, 'status': 'idle'}

    async def start(self, parent_task_id: str, goal_description: str):
        """Start the Team Lead loop to decompose goals and delegate tasks."""
        self.state['parent_task_id'] = parent_task_id
        self.state['status'] = 'pending_delegation'
        subtask = self.decomposeTaskStub(goal_description)
        target_agent_id = self.selectMemberStub()
        self.state['status'] = 'delegating'
        # Invoke delegate_task tool
        await self.delegate_tool(
            target_agent_id=target_agent_id,
            task_description=subtask['description'],
            parent_task_id=self.state['parent_task_id'],
            calling_agent_id=self.agent_id,
            calling_team=self.team_id
        )
        self.state['status'] = 'delegated'

    def decomposeTaskStub(self, goal_description: str) -> dict:
        """Placeholder for task decomposition stub."""
        import uuid
        return {
            'id': f"subtask_{uuid.uuid4().hex[:8]}",
            'description': f"Perform initial analysis based on goal: {goal_description}"
        }

    def selectMemberStub(self) -> str:
        """Placeholder for selecting a team member stub."""
        # Select the first available member by default
        return self.member_ids[0] if self.member_ids else ''
