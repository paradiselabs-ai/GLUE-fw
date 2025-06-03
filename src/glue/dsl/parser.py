# src/glue/dsl/parser.py

import logging
from lark import Lark, Transformer, v_args

# ==================== Constants ===================='
logger = logging.getLogger("glue.dsl.stickyscript")

STICKY_SCRIPT_GRAMMAR = r"""
    start: declarations*

    declarations: app_decl | agent_decl | tool_decl | team_decl | flow_decl | glue_app_decl | model_decl

    app_decl: "app" CNAME (_LBRACE app_options? _RBRACE)?
    glue_app_decl: "glue" "app" (_LBRACE app_options? _RBRACE)?
    app_options: app_option+
    app_option: "description" ASSIGNMENT STRING -> app_description
             | "name" ASSIGNMENT STRING -> app_name
             | "config" _LBRACE config_options _RBRACE -> app_config

    config_options: config_option+
    config_option: CNAME ASSIGNMENT (STRING | CNAME | NUMBER) -> config_item

    model_decl: "model" CNAME _LBRACE model_options _RBRACE
    model_options: model_option+
    model_option: "provider" ASSIGNMENT STRING -> model_provider
               | "role" ASSIGNMENT STRING -> model_role
               | "adhesives" ASSIGNMENT _LBRACKET cname_list? _RBRACKET -> model_adhesives
               | "config" _LBRACE config_options _RBRACE -> model_config

    agent_decl: "agent" CNAME _LBRACE agent_options _RBRACE
    agent_options: agent_option+
    agent_option: "model" ASSIGNMENT STRING -> agent_model
                | "provider" ASSIGNMENT STRING -> agent_provider
                | "instructions" ASSIGNMENT STRING -> agent_instructions
                | "role" ASSIGNMENT STRING -> agent_role
                | "adhesives" ASSIGNMENT _LBRACKET cname_list? _RBRACKET -> agent_adhesives
                | "config" _LBRACE config_options _RBRACE -> agent_config

    tool_decl: "tool" CNAME (_LBRACE tool_options? _RBRACE)?
    tool_options: tool_option+
    tool_option: "description" ASSIGNMENT STRING -> tool_description
               | "provider" ASSIGNMENT CNAME -> tool_provider
               | "config" _LBRACE config_options _RBRACE -> tool_config

    team_decl: "team" CNAME _LBRACE team_options _RBRACE
    team_options: team_option+
    team_option: "lead" ASSIGNMENT CNAME -> team_lead
               | "members" ASSIGNMENT _LBRACKET cname_list? _RBRACKET -> team_members
               | "tools" ASSIGNMENT _LBRACKET cname_list? _RBRACKET -> team_tools
               | "instructions" ASSIGNMENT STRING -> team_instructions

    cname_list: CNAME (COMMA CNAME)*

    flow_decl: "flow" CNAME ARROW CNAME -> basic_flow

    _LBRACE: "{"
    _RBRACE: "}"
    _LBRACKET: "["
    _RBRACKET: "]"
    COMMA: ","
    COLON: ":"
    ASSIGNMENT: ":" | "="
    ARROW: "->"

    STRING: /\"(?:\\.|[^\"\\])*\"/ | /\'(?:\\.|[^\'\\])*\'/ // Improved STRING to handle escaped quotes
    NUMBER: /\d+(\.\d+)?/

    %import common.CNAME
    %import common.WS
    %ignore WS
"""

class TreeToDict(Transformer):
    def __init__(self):
        super().__init__()
        # Initialize config in transform method or ensure it's reset for each call
        # For simplicity here, we'll rely on a new instance per parse in StickyScriptParser

    def _init_config(self):
        return {
            "workflow": {"name": "DefaultGlueApp"},
            "agents": {},
            "teams": {},
            "tools": {},
            "models": {},
            "flows": {}
        }

    def STRING(self, s):
        # Remove quotes from the string
        return s[1:-1]

    def CNAME(self, token):
        return token.value
        
    def NUMBER(self, token):
        # Convert to int or float as appropriate
        value = token.value
        if '.' in value:
            return float(value)
        return int(value)

    def cname_list(self, items):
        return list(items)

    def app_decl(self, items):
        app_name = items[0]
        self.config["workflow"]["name"] = app_name
        return None

    def glue_app_decl(self, items):
        # Default app name is already set in _init_config
        return None

    def app_description(self, items):
        description = items[0]
        self.config["workflow"]["description"] = description
        return None
        
    def app_name(self, items):
        name = items[0]
        self.config["workflow"]["name"] = name
        return None
        
    def config_item(self, items):
        key = items[0]
        value = items[1]
        return (key, value)
        
    def config_options(self, items):
        options = {}
        for item in items:
            if item is not None and isinstance(item, tuple):
                key, value = item
                options[key] = value
        return options
        
    def app_config(self, items):
        config = items[0] if items else {}
        self.config["workflow"]["config"] = config
        return None

    def agent_model(self, items):
        return ("model", items[0])

    def agent_provider(self, items):
        return ("provider", items[0])

    def agent_instructions(self, items):
        return ("instructions", items[0])
        
    def agent_role(self, items):
        return ("role", items[0])
        
    def agent_adhesives(self, items):
        adhesives = items[1] if len(items) > 1 else []
        return ("adhesives", adhesives)
        
    def agent_config(self, items):
        return ("config", items[0] if items else {})
        
    def model_provider(self, items):
        return ("provider", items[0])
        
    def model_role(self, items):
        return ("role", items[0])
        
    def model_adhesives(self, items):
        adhesives = items[1] if len(items) > 1 else []
        return ("adhesives", adhesives)
        
    def model_config(self, items):
        return ("config", items[0] if items else {})
        
    def model_options(self, items):
        options = {}
        for item in items:
            if item is not None and isinstance(item, tuple):
                key, value = item
                options[key] = value
        return options
        
    def model_decl(self, items):
        model_name = items[0]
        model_options = items[1] if len(items) > 1 else {}
        self.config["models"][model_name] = model_options
        return None

    def agent_options(self, items):
        # Flatten the list of tuples into a dictionary
        options = {}
        for item in items:
            if item is not None and isinstance(item, tuple):
                key, value = item
                options[key] = value
        return options

    def agent_decl(self, items):
        agent_name = items[0]
        agent_options = items[1] if len(items) > 1 else {}
        self.config["agents"][agent_name] = agent_options
        return None

    def tool_description(self, items):
        return ("description", items[0])
        
    def tool_provider(self, items):
        return ("provider", items[0])
        
    def tool_config(self, items):
        return ("config", items[0] if items else {})

    def tool_options(self, items):
        # Similar to agent_options
        options = {}
        for item in items:
            if item is not None and isinstance(item, tuple):
                key, value = item
                options[key] = value
        return options

    def tool_decl(self, items):
        tool_name = items[0]
        tool_options = items[1] if len(items) > 1 else {}
        self.config["tools"][tool_name] = tool_options
        return None

    def team_lead(self, items):
        return ("lead", items[0])

    def team_members(self, items):
        members = items[1] if len(items) > 1 else []
        return ("members", members)

    def team_tools(self, items):
        tools = items[1] if len(items) > 1 else []
        return ("tools", tools)

    def team_instructions(self, items):
        return ("instructions", items[0])

    def team_options(self, items):
        options = {"members": [], "tools": []} # Initialize with defaults
        for item in items:
            if item is not None and isinstance(item, tuple):
                key, value = item
                options[key] = value
        return options

    def team_decl(self, items):
        team_name = items[0]
        team_options = items[1] if len(items) > 1 else {}
        # Ensure defaults if not set
        if "members" not in team_options:
            team_options["members"] = []
        if "tools" not in team_options:
            team_options["tools"] = []
        self.config["teams"][team_name] = team_options
        return None

    # Flow declarations
    def basic_flow(self, items):
        # items = [from_CNAME_str, ARROW_token, to_CNAME_str]
        from_team = items[0]
        to_team = items[2] 
        flow_name = f"flow_{self._flow_counter}"
        self.config["flows"][flow_name] = {
            "from": from_team,
            "to": to_team,
            "type": "basic"
        }
        self._flow_counter += 1
        return items
        
    def declarations(self, items):
        # This rule helps group all top-level declarations.
        # No specific action needed here as children rules modify self.config.
        return items

    def start(self, items):
        # This is the entry point for the transformation.
        # items is a list of results from 'declarations*'
        # The config should be built up by child rules.
        if not self.config["workflow"].get("name"):
            self.config["workflow"]["name"] = "DefaultGlueApp"
        # _current_xxx_name attributes are no longer used.
        return self.config

    def transform(self, tree):
        # Reset or initialize state for each transformation
        self.config = self._init_config()
        self._flow_counter = 0
        return super().transform(tree)


class StickyScriptParser:
    def __init__(self):
        # Initialize Lark parser. Transformer instance is passed on parse.
        self.parser = Lark(STICKY_SCRIPT_GRAMMAR, parser='lalr', start='start')
        logger.info("StickyScriptParser initialized with grammar.")

    def parse(self, script_text: str):
        logger.debug(f"Attempting to parse script:\n{script_text}")
        try:
            transformer_instance = TreeToDict()
            tree = self.parser.parse(script_text)
            # The transformer_instance will build up its internal .config dict
            # and the 'start' method will return this dict.
            parsed_config = transformer_instance.transform(tree)
            
            logger.debug(f"Successfully parsed. Resulting config: {parsed_config}")
            return parsed_config
        except Exception as e:
            logger.error(f"Error parsing StickyScript: {e}", exc_info=True)
            # Return a consistent error structure
            return {
                "error": str(e),
                "workflow": {"name": "ErrorParsingApp"},
                "agents": {}, "teams": {}, "tools": {}, "flows": {}
            }

    def parse_file(self, file_path: str):
        """Parse a StickyScript file from the given path.
        
        Args:
            file_path: Path to the StickyScript file
            
        Returns:
            Parsed configuration dictionary
        """
        logger.debug(f"Attempting to parse file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                script_text = f.read()
            return self.parse(script_text)
        except Exception as e:
            logger.error(f"Error reading or parsing file {file_path}: {e}", exc_info=True)
            # Return a consistent error structure
            return {
                "error": str(e),
                "workflow": {"name": "ErrorParsingFile"},
                "agents": {}, "teams": {}, "tools": {}, "flows": {}
            }

# Example Usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = StickyScriptParser()

    print("\n--- Testing App Declarations ---")
    script_app_simple = "app MyApp"
    print(f"Parsing: {script_app_simple}\nResult: {parser.parse(script_app_simple)}")

    script_app_desc = 'app AnotherApp { description: "A cool app with \\"quotes\\"." }'
    print(f"\nParsing: {script_app_desc}\nResult: {parser.parse(script_app_desc)}")

    print("\n--- Testing Agent Declarations ---")
    script_agent = '''
    agent MyAgent {
        model: "gpt-4-turbo",
        provider: "openai",
        instructions: "You are an advanced AI assistant."
    }
    '''
    print(f"Parsing agent script...\\nResult: {parser.parse(script_agent)}")

    print("\\n--- Testing Tool Declarations ---")
    script_tool_simple = "tool MySimpleTool"
    print(f"Parsing: {script_tool_simple}\\nResult: {parser.parse(script_tool_simple)}")
    
    script_tool_desc = 'tool MyComplexTool { description: "Performs intricate operations." }'
    print(f"Parsing: {script_tool_desc}\\nResult: {parser.parse(script_tool_desc)}")

    print("\\n--- Testing Team Declarations ---")
    script_team = '''
    agent LeaderBot { model: "claude-3-opus", provider: "anthropic", instructions: "Strategize and lead."}
    agent WorkerBot1 { model: "gemini-1.5-pro", provider: "google", instructions: "Execute tasks efficiently."}
    tool UtilityTool
    
    team EngineeringTeam {
        lead: LeaderBot,
        members: [WorkerBot1],
        tools: [UtilityTool],
        instructions: "Build innovative solutions."
    }
    '''
    print(f"Parsing team script...\\nResult: {parser.parse(script_team)}")
    
    print("\\n--- Testing Flow Declarations ---")
    # Need to declare agents and teams first for flows to be meaningful in a full script
    script_flow_setup = """
    app MyWorkflow
    agent AgentX { model: "m", provider: "p", instructions: "i" }
    agent AgentY { model: "m", provider: "p", instructions: "i" }
    team TeamAlpha { lead: AgentX }
    team TeamBeta { lead: AgentY }
    """
    script_flow_actual = "flow TeamAlpha -> TeamBeta"
    
    # Test parsing them together
    full_flow_script = script_flow_setup + script_flow_actual
    print(f"Parsing flow script...\\nResult: {parser.parse(full_flow_script)}")


    print("\\n--- Testing Combined Script ---")
    combined_script = """
    app MyFullApp {
        description: "A comprehensive application demonstration."
    }

    agent Planner {
        model: "gpt-4o",
        provider: "openai",
        instructions: "Formulate detailed plans."
    }

    agent Coder {
        model: "claude-3-sonnet",
        provider: "anthropic",
        instructions: "Write clean and efficient code."
    }
    
    tool SearchTool {
        description: "Tool for searching documents."
    }

    team DevelopmentTeam {
        lead: Planner,
        members: [Coder],
        tools: [SearchTool],
        instructions: "Develop features as per plan."
    }
    
    flow DevelopmentTeam -> Planner // Example of a feedback loop or different flow
    """
    print(f"Parsing combined script...\\nResult: {parser.parse(combined_script)}")

    print("\\n--- Testing Empty Script ---")
    empty_script = ""
    print(f"Parsing empty script...\\nResult: {parser.parse(empty_script)}")
    
    print("\\n--- Testing Script with only comments (ignored by grammar) ---")
    # Note: Lark by default doesn't have a concept of line comments like //
    # unless specified in grammar or via a pre-processing step.
    # The %ignore WS handles whitespace, but not comments unless defined.
    # For this test, we'll assume comments are stripped or handled if grammar supports it.
    # Current grammar does not explicitly support `//` comments.
    # If `//` comments are needed, grammar should be:
    # COMMENT: "//" /[^\\n]*/ 
    # %ignore COMMENT
    # For now, this test will likely parse as empty or error if comments aren't ignored.
    comment_script = """
    // app MyCommentedApp 
    // This should be ignored
    """ 
    print(f"Parsing script with comments (current grammar may not ignore '//')...\\nResult: {parser.parse(comment_script)}")

    script_multiple_agents_teams = """
    app MultiTest
    agent A1 {model:"m1", provider:"p1", instructions:"i1"}
    agent A2 {model:"m2", provider:"p2", instructions:"i2"}
    tool T1
    tool T2 {description: "d2"}
    team Alpha {lead: A1, members: [A2], tools: [T1, T2], instructions: "Alpha team"}
    team Beta {lead: A2, members: [A1], tools: [T1], instructions: "Beta team"}
    flow Alpha -> Beta
    flow Beta -> Alpha
    """
    print(f"\\nParsing multiple entities script...\\nResult: {parser.parse(script_multiple_agents_teams)}")
