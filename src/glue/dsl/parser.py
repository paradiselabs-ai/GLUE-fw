# glue/dsl/parser.py
# ==================== Imports ====================
from typing import Dict, List, Any
from pathlib import Path
import logging

# Import the lexer and token types from their respective modules
from .tokens import TokenType, Token
from .lexer import GlueLexer

# Temporarily comment out imports that might not exist yet
# from ..core.schemas import ModelConfig, TeamConfig, ToolConfig
# from ..magnetic.field import FlowType

# ==================== Constants ====================
logger = logging.getLogger("glue.dsl")


class GlueParser:
    """Parser for the GLUE Domain Specific Language.

    This class parses tokens into an Abstract Syntax Tree (AST).
    """

    def __init__(self, tokens: List[Token] = None, schema_version: str = "1.0"):
        """Initialize a new GlueParser"""
        self.tokens = tokens or []
        self.pos = 0
        self.ast = {
            "app": {},
            "teams": [],
            "models": {},
            "tools": {},
            "flows": [],
            "magnetize": {},
            "apply": None,
        }
        self.schema_version = schema_version
        self.logger = logging.getLogger("glue.dsl.parser")

    def parse(self, tokens: List[Token] = None) -> Dict[str, Any]:
        """Parse tokens into AST"""
        if tokens is not None:
            self.tokens = tokens
            self.pos = 0

        # Reset AST
        self.ast = {
            "app": {},
            "teams": [],
            "models": {},
            "tools": {},
            "flows": [],
            "magnetize": {},
            "apply": None,
        }

        # Parse tokens
        while not self._is_at_end():
            if self._check(TokenType.KEYWORD):
                keyword = self._peek().value
                if keyword == "glue":
                    self._parse_app()
                elif keyword == "model":
                    self._parse_model()
                elif keyword == "tool":
                    self._parse_tool()
                elif keyword == "magnetize":
                    self._parse_magnetize()
                elif keyword == "apply":
                    self._parse_apply()
                else:
                    raise SyntaxError(
                        f"Unexpected keyword at line {self._peek().line}: {keyword}"
                    )
            elif self._check(TokenType.APPLY_GLUE):
                # Handle 'apply_glue' token
                self._advance()  # Consume the token
                self.ast["apply"] = "glue"
            elif self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
            else:
                token = self._peek()
                raise SyntaxError(
                    f"Unexpected token at line {token.line}: {token.type}"
                )

        return self.ast

    def _parse_app(self):
        """Parse app configuration"""
        # Expect 'glue' keyword
        self._consume(TokenType.KEYWORD, "Expected 'glue' keyword")

        # Expect 'app' keyword
        self._consume(TokenType.KEYWORD, "Expected 'app' keyword")

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'app'")

        # Parse app properties
        app_props = {}
        self._parse_properties(app_props)

        # Store app properties in AST
        self.ast["app"] = app_props

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after app properties")

    def _parse_model(self):
        """Parse model definition"""
        # Expect 'model' keyword
        self._consume(TokenType.KEYWORD, "Expected 'model' keyword")

        # Expect model name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected model name")
        model_name = name_token.value
        model = {}

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after model name")

        # Parse model properties
        self._parse_properties(model)

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after model properties")

        # Add model to AST
        self.ast["models"][model_name] = model

    def _parse_tool(self):
        """Parse tool definition"""
        # Expect 'tool' keyword
        self._consume(TokenType.KEYWORD, "Expected 'tool' keyword")

        # Expect tool name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected tool name")
        tool_name = name_token.value
        tool = {}

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after tool name")

        # Parse tool properties
        self._parse_properties(tool)

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after tool properties")

        # Add tool to AST
        self.ast["tools"][tool_name] = tool

    def _parse_magnetize(self):
        """Parse magnetic field configuration"""
        # Expect 'magnetize' keyword
        self._consume(TokenType.KEYWORD, "Expected 'magnetize' keyword")

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'magnetize'")

        # Parse teams and flow section
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.KEYWORD):
                keyword = self._peek().value
                if keyword == "team":
                    self._parse_team()
                elif keyword == "flow":
                    # Handle flow section inside magnetize block
                    self._parse_flow()
                else:
                    raise SyntaxError(
                        f"Unexpected keyword at line {self._peek().line}: {keyword}"
                    )
            elif self._check(TokenType.IDENTIFIER):
                # Parse team defined by identifier
                team_name = self._advance().value
                team = {}

                # Expect opening brace
                self._consume(
                    TokenType.LBRACE, f"Expected '{{' after team name '{team_name}'"
                )

                # Parse team properties
                self._parse_properties(team)

                # Expect closing brace
                self._consume(
                    TokenType.RBRACE,
                    f"Expected '}}' after team '{team_name}' properties",
                )

                # Add team to magnetize section
                self.ast["magnetize"][team_name] = team
            elif self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
            else:
                token = self._peek()
                raise SyntaxError(
                    f"Unexpected token at line {token.line}: {token.type}"
                )

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after magnetize block")

    def _parse_team(self):
        """Parse team definition within magnetize block"""
        # Expect team name
        name_token = self._consume(TokenType.KEYWORD, "Expected 'team' keyword")
        if name_token.value != "team":
            raise SyntaxError(
                f"Expected 'team' keyword at line {name_token.line}, got '{name_token.value}'"
            )
        name_token = self._consume(TokenType.IDENTIFIER, "Expected team name")
        team = {}

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after team name")

        # Parse team properties
        self._parse_properties(team)

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after team properties")

        # Add team to magnetize section
        self.ast["magnetize"][name_token.value] = team

    def _parse_flow(self):
        """Parse flow section within magnetize block"""
        # Expect 'flow' keyword
        self._consume(TokenType.KEYWORD, "Expected 'flow' keyword")

        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'flow'")

        # Parse flow definitions
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
                continue

            # Expect source team identifier
            if not self._check(TokenType.IDENTIFIER):
                token = self._peek()
                raise SyntaxError(
                    f"Expected source team identifier at line {token.line}, got {token.type}"
                )

            source_token = self._advance()
            source_team = source_token.value

            # Expect flow operator
            flow_type = None
            if not self._check(TokenType.ARROW):
                token = self._peek()
                raise SyntaxError(
                    f"Expected flow operator at line {token.line}, got {token.type}"
                )

            arrow_token = self._advance()
            if arrow_token.value == "->":
                flow_type = "PUSH"
            elif arrow_token.value == "><":
                flow_type = "BIDIRECTIONAL"
            elif arrow_token.value == "<>":
                flow_type = "REPEL"
            elif arrow_token.value == "<-":
                flow_type = "PULL"
            else:
                raise SyntaxError(
                    f"Unknown flow operator at line {arrow_token.line}: {arrow_token.value}"
                )

            # Expect target team identifier
            if not self._check(TokenType.IDENTIFIER):
                token = self._peek()
                raise SyntaxError(
                    f"Expected target team identifier at line {token.line}, got {token.type}"
                )

            target_token = self._advance()
            target_team = target_token.value

            # Create flow definition
            flow = {"source": source_team, "target": target_team, "type": flow_type}

            # Add flow to AST
            self.ast["flows"].append(flow)

            # Check for semicolon or comment
            if self._check(TokenType.SEMICOLON):
                self._advance()
            elif self._check(TokenType.COMMENT):
                self._advance()

        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after flow definitions")

    def _parse_apply(self):
        """Parse apply statement"""
        # Expect 'apply' keyword
        self._consume(TokenType.KEYWORD, "Expected 'apply' keyword")

        # Expect 'glue' keyword
        self._consume(TokenType.KEYWORD, "Expected 'glue' keyword")

        # No properties for apply glue statement
        self.ast["apply"] = "glue"

    def _parse_properties(self, target: Dict[str, Any]):
        """Parse properties into the target dictionary"""
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
                continue

            # Expect property name (can be identifier or keyword)
            if self._check(TokenType.IDENTIFIER):
                name_token = self._advance()
            elif self._check(TokenType.KEYWORD):
                name_token = self._advance()
            else:
                raise SyntaxError(
                    f"Expected property name at line {self._peek().line}, got {self._peek().type}"
                )

            property_name = name_token.value

            # Special handling for 'tools' property in team definitions
            if property_name == "tools":
                self._consume(TokenType.EQUALS, f"Expected '=' after {property_name}")
                tools = self._parse_value()
                if not isinstance(tools, list):
                    raise SyntaxError(
                        f"Expected list of tools at line {name_token.line}, got {type(tools).__name__}"
                    )
                target[property_name] = tools
                continue

            # Special handling for 'tool' property in team definitions (singular form)
            if property_name == "tool" and self._check(TokenType.IDENTIFIER):
                # This is a special case for 'tool s' in the magnetize block
                if self._peek().value == "s":
                    self._advance()  # Consume 's'
                    self._consume(
                        TokenType.EQUALS, f"Expected '=' after {property_name}s"
                    )
                    tools = self._parse_value()
                    if not isinstance(tools, list):
                        raise SyntaxError(
                            f"Expected list of tools at line {name_token.line}, got {type(tools).__name__}"
                        )
                    target["tools"] = tools
                    continue

            # Special handling for 'adhesives' property in model definitions
            if property_name == "adhesives":
                self._consume(TokenType.EQUALS, f"Expected '=' after {property_name}")
                adhesives = self._parse_value()
                if not isinstance(adhesives, list):
                    raise SyntaxError(
                        f"Expected list of adhesives at line {name_token.line}, got {type(adhesives).__name__}"
                    )
                target[property_name] = adhesives
                continue

            # Check for nested object (like config block)
            if self._check(TokenType.LBRACE):
                # Parse nested object
                self._advance()  # Consume '{'

                if property_name not in target:
                    target[property_name] = {}

                # Create a temporary dictionary for the nested properties
                nested_properties = {}
                self._parse_properties(nested_properties)

                # Process any special values in the nested properties
                for key, value in nested_properties.items():
                    # Convert string boolean values to actual booleans
                    if isinstance(value, str):
                        if value.lower() == "true":
                            nested_properties[key] = True
                        elif value.lower() == "false":
                            nested_properties[key] = False

                # Assign the processed nested properties to the target
                target[property_name] = nested_properties

                self._consume(
                    TokenType.RBRACE, f"Expected '}}' after {property_name} properties"
                )
            else:
                # Expect equals sign
                self._consume(TokenType.EQUALS, f"Expected '=' after {property_name}")

                # Parse value
                value = self._parse_value()

                # Special handling for string values - remove quotes
                if (
                    isinstance(value, str)
                    and value.startswith('"')
                    and value.endswith('"')
                ):
                    value = value[1:-1]

                # Add property to target
                target[property_name] = value

    def _parse_value(self) -> Any:
        """Parse a value (string, number, boolean, array, or identifier)"""
        if self._check(TokenType.STRING):
            return self._advance().value
        elif self._check(TokenType.NUMBER):
            # Convert to float or int as appropriate
            value = self._advance().value
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        elif self._check(TokenType.BOOLEAN):
            # Handle boolean tokens
            value = self._advance().value
            if value.lower() == "true":
                return True
            elif value.lower() == "false":
                return False
            else:
                return value
        elif self._check(TokenType.KEYWORD):
            keyword = self._advance().value
            # Handle boolean literals
            if keyword == "true":
                return True
            elif keyword == "false":
                return False
            else:
                return keyword
        elif self._check(TokenType.IDENTIFIER):
            return self._advance().value
        elif self._check(TokenType.LBRACKET):
            return self._parse_array()
        else:
            raise SyntaxError(
                f"Expected value at line {self._peek().line}, got {self._peek().type}"
            )

    def _parse_array(self) -> List[Any]:
        """Parse an array"""
        array = []

        # Expect opening bracket
        self._consume(TokenType.LBRACKET, "Expected '['")

        # Parse array elements
        while not self._check(TokenType.RBRACKET) and not self._is_at_end():
            value = self._parse_value()
            array.append(value)

            if self._check(TokenType.RBRACKET):
                break

            self._consume(TokenType.COMMA, "Expected ',' between array elements")

        # Expect closing bracket
        self._consume(TokenType.RBRACKET, "Expected ']' after array")
        return array

    def _advance(self) -> Token:
        """Advance to the next token and return the current one"""
        if not self._is_at_end():
            self.pos += 1
        return self.tokens[self.pos - 1]

    def _peek(self) -> Token:
        """Return the current token without advancing"""
        if self._is_at_end():
            return self.tokens[-1]  # Return EOF token
        return self.tokens[self.pos]

    def _is_at_end(self) -> bool:
        """Check if we've reached the end of the token stream"""
        return (
            self.pos >= len(self.tokens) or self.tokens[self.pos].type == TokenType.EOF
        )

    def _check(self, type_: TokenType) -> bool:
        """Check if the current token is of the given type"""
        if self._is_at_end():
            return False
        return self.tokens[self.pos].type == type_

    def _consume(self, type_: TokenType, error_message: str) -> Token:
        """Consume a token of the expected type or raise an error"""
        # Special handling for keywords that can be either KEYWORD or IDENTIFIER
        if type_ == TokenType.KEYWORD and self._check(TokenType.IDENTIFIER):
            # If we're expecting a keyword but have an identifier, check if it's a keyword value
            current = self._peek()
            if current.value in [
                "app",
                "model",
                "tool",
                "apply",
                "tools",
                "glue",
                "config",
                "magnetize",
                "team",
                "provider",
                "role",
                "adhesives",
                "lead",
            ]:
                return self._advance()

        if self._check(type_):
            return self._advance()

        current = self._peek()
        raise SyntaxError(f"{error_message} at line {current.line}, got {current.type}")

    def validate(self, config: Dict[str, Any] = None) -> List[str]:
        """Validate a parsed GLUE configuration.

        Args:
            config: Parsed GLUE configuration, uses self.ast if None

        Returns:
            List of validation errors, empty if valid
        """
        self.logger.info("Validating GLUE configuration")

        if config is None:
            config = self.ast

        errors = []

        # Check for required sections
        if "app" not in config or not config["app"]:
            errors.append("Missing 'app' section")

        # Validate app section
        if "app" in config and config["app"]:
            app_config = config["app"]
            if "name" not in app_config:
                errors.append("Missing 'name' in app configuration")

        return errors

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a GLUE DSL file.

        Args:
            file_path: Path to the GLUE DSL file

        Returns:
            Parsed configuration as a dictionary

        Raises:
            FileNotFoundError: If the file does not exist
            SyntaxError: If the file contains syntax errors
            ValueError: If the file contains semantic errors
        """
        self.logger.info(f"Parsing GLUE file: {file_path}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix != ".glue":
            raise ValueError("File must have .glue extension")

        with path.open("r") as f:
            content = f.read()

        return self.parse_string(content)

    def parse_string(self, content: str) -> Dict[str, Any]:
        """Parse GLUE DSL content from a string.

        Args:
            content: GLUE DSL content as a string

        Returns:
            Parsed configuration as a dictionary

        Raises:
            SyntaxError: If the content contains syntax errors
            ValueError: If the content contains semantic errors
        """
        self.logger.info("Parsing GLUE content from string")

        # Tokenize
        lexer = GlueLexer()
        tokens = lexer.tokenize(content)

        # Parse
        return self.parse(tokens)


# For backward compatibility
GlueDSLParser = GlueParser
