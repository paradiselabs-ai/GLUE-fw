from typing import Any, Dict

class GlueSmolTool:
    """
    A wrapper that adapts a GLUE tool to the Smolagents tool interface,
    adding adhesive persistence and permission logic.
    """
    def __init__(self, glue_tool: Any):
        self._glue_tool = glue_tool
        self.name = glue_tool.name
        self.description = getattr(glue_tool, 'description', '') or glue_tool.name
        # Map GLUE inputs schema to Smolagents input schema with JSON type strings
        raw_inputs = getattr(glue_tool, 'inputs', {}) or {}
        self.inputs = {}
        for name, info in raw_inputs.items():
            raw_type = info.get('type')
            if raw_type is str:
                type_str = 'string'
            elif raw_type is int:
                type_str = 'integer'
            elif raw_type is float:
                type_str = 'number'
            elif raw_type is bool:
                type_str = 'boolean'
            elif raw_type is list:
                type_str = 'array'
            elif raw_type is dict:
                type_str = 'object'
            else:
                type_str = raw_type.lower() if isinstance(raw_type, str) else 'any'
            self.inputs[name] = {
                'type': type_str,
                'description': info.get('description', '')
            }
        # Map GLUE output type for Smolagents
        raw_out = getattr(glue_tool, 'output_type', 'Any')
        if isinstance(raw_out, str):
            self.output_type = raw_out.lower()
        else:
            if raw_out is str:
                self.output_type = 'string'
            elif raw_out is int:
                self.output_type = 'integer'
            elif raw_out is float:
                self.output_type = 'number'
            elif raw_out is bool:
                self.output_type = 'boolean'
            else:
                self.output_type = 'any'

    def __call__(self, **kwargs: Any) -> Any:
        """
        Invoke the wrapped GLUE tool with potential adhesive logic.
        """
        # Alias mapping for LLM-friendly argument names
        if "target_name" in kwargs:
            kwargs["target_agent_id"] = kwargs.pop("target_name")
        if "task" in kwargs:
            kwargs["task_description"] = kwargs.pop("task")
        # Pre-execution: apply any GLUE adhesive or permission checks
        # TODO: insert pre-tool execution glue hooks

        result = self._glue_tool.execute(**kwargs)

        # Post-execution: handle adhesive persistence if configured
        # TODO: insert post-tool execution glue hooks

        return result 