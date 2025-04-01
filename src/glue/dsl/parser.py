# glue/dsl/parser.py
# ==================== Imports ====================
from enum import Enum
from typing import Dict, List, Any, Optional, Set
import re
from dataclasses import dataclass
from pathlib import Path
import logging

# Temporarily comment out imports that might not exist yet
# from ..core.types import ModelConfig, TeamConfig, ToolConfig
# from ..magnetic.field import FlowType

# ==================== Constants ====================
logger = logging.getLogger("glue.dsl")

class TokenType(Enum):
    """Token types for DSL parsing"""
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    EQUALS = "EQUALS"
    LBRACE = "LBRACE"
    RBRACE = "RBRACE"
    LBRACKET = "LBRACKET"
    RBRACKET = "RBRACKET"
    COMMA = "COMMA"
    ARROW = "ARROW"      # For magnetic operators
    SEMICOLON = "SEMICOLON"
    COMMENT = "COMMENT"
    EOF = "EOF"

# ==================== Class Definitions ====================
@dataclass
class Token:
    """Token representation"""
    type: TokenType
    value: str
    line: int

class GlueLexer:
    """Lexical analyzer for GLUE DSL"""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.tokens = []
        
    def tokenize(self) -> List[Token]:
        """Convert source into tokens"""
        while self.pos < len(self.source):
            char = self.source[self.pos]
            
            # Skip whitespace
            if char.isspace():
                if char == '\n':
                    self.line += 1
                self.pos += 1
                continue
                
            # Handle comments
            if char == '/' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '/':
                self._tokenize_comment()
                continue
                
            # Handle identifiers
            if char.isalpha() or char == '_':
                self._tokenize_identifier()
                continue
                
            # Handle strings
            if char in ('"', "'"):
                self._tokenize_string()
                continue
                
            # Handle numbers
            if char.isdigit():
                self._tokenize_number()
                continue
                
            # Handle operators and delimiters
            if char == '=':
                self.tokens.append(Token(TokenType.EQUALS, '=', self.line))
                self.pos += 1
                continue
                
            if char == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', self.line))
                self.pos += 1
                continue
                
            if char == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', self.line))
                self.pos += 1
                continue
                
            if char == '[':
                self.tokens.append(Token(TokenType.LBRACKET, '[', self.line))
                self.pos += 1
                continue
                
            if char == ']':
                self.tokens.append(Token(TokenType.RBRACKET, ']', self.line))
                self.pos += 1
                continue
                
            if char == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', self.line))
                self.pos += 1
                continue
                
            if char == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', self.line))
                self.pos += 1
                continue
                
            # Handle magnetic operators
            if char in ('<', '>', '-'):
                self._tokenize_magnetic_operator()
                continue
                
            # Unrecognized character
            self.pos += 1
            
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line))
        return self.tokens
        
    def _tokenize_identifier(self):
        """Tokenize an identifier"""
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.pos += 1
            
        value = self.source[start:self.pos]
        
        # Check for boolean values
        if value.lower() == 'true':
            self.tokens.append(Token(TokenType.BOOLEAN, 'true', self.line))
        elif value.lower() == 'false':
            self.tokens.append(Token(TokenType.BOOLEAN, 'false', self.line))
        else:
            self.tokens.append(Token(TokenType.IDENTIFIER, value, self.line))
            
    def _tokenize_string(self):
        """Tokenize a string literal"""
        delimiter = self.source[self.pos]
        self.pos += 1  # Skip opening quote
        start = self.pos
        
        # Handle escaped characters and quotes
        escape_mode = False
        while self.pos < len(self.source):
            if escape_mode:
                # Skip the escaped character
                escape_mode = False
            elif self.source[self.pos] == '\\':
                escape_mode = True
            elif self.source[self.pos] == delimiter and not escape_mode:
                break
            
            if self.source[self.pos] == '\n':
                self.line += 1
            
            self.pos += 1
            
        if self.pos >= len(self.source):
            raise ValueError(f"Unterminated string at line {self.line}")
            
        value = self.source[start:self.pos]
        self.tokens.append(Token(TokenType.STRING, value, self.line))
        self.pos += 1  # Skip closing quote
        
    def _tokenize_number(self):
        """Tokenize a number literal"""
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            self.pos += 1
            
        value = self.source[start:self.pos]
        self.tokens.append(Token(TokenType.NUMBER, value, self.line))
        
    def _tokenize_magnetic_operator(self):
        """Tokenize magnetic field operators"""
        start = self.pos
        # Look ahead for full operator
        while self.pos < len(self.source) and self.source[self.pos] in ('<', '>', '-'):
            self.pos += 1
            
        operator = self.source[start:self.pos]
        
        # Validate operator
        if operator in ("->", "<-", "><", "<>"):
            self.tokens.append(Token(TokenType.ARROW, operator, self.line))
        else:
            raise ValueError(f"Invalid magnetic operator at line {self.line}: {operator}")
            
    def _tokenize_comment(self):
        """Skip comment lines"""
        self.pos += 2  # Skip //
        while self.pos < len(self.source) and self.source[self.pos] != '\n':
            self.pos += 1
            
        if self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.line += 1
            self.pos += 1

class GlueParser:
    """Parser for GLUE DSL"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.ast = {
            "app": {},
            "teams": [],
            "models": [],
            "tools": [],
            "flows": []
        }
        
    def parse(self) -> Dict[str, Any]:
        """Parse tokens into AST"""
        while not self._is_at_end():
            token = self._peek()
            
            if token.type == TokenType.IDENTIFIER:
                if token.value == "glue":
                    self._parse_app()
                elif token.value == "model":
                    self._parse_model()
                elif token.value == "tool":
                    self._parse_tool()
                elif token.value == "magnetize":
                    self._parse_magnetize()
                else:
                    raise SyntaxError(f"Unexpected identifier at line {token.line}: {token.value}")
            elif token.type == TokenType.COMMENT:
                # Skip comments
                self._advance()
            else:
                raise SyntaxError(f"Unexpected token at line {token.line}: {token.type}")
                
        return self.ast
        
    def _parse_app(self):
        """Parse app configuration"""
        # Expect 'glue' identifier
        self._consume(TokenType.IDENTIFIER, "Expected 'glue' keyword")
        
        # Expect 'app' identifier
        app_token = self._consume(TokenType.IDENTIFIER, "Expected 'app' keyword")
        if app_token.value != "app":
            raise SyntaxError(f"Expected 'app' keyword at line {app_token.line}, got '{app_token.value}'")
            
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'app'")
        
        # Parse app properties
        self._parse_properties(self.ast["app"])
        
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after app properties")
        
    def _parse_model(self):
        """Parse model definition"""
        # Expect 'model' identifier
        self._consume(TokenType.IDENTIFIER, "Expected 'model' keyword")
        
        # Expect model name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected model name")
        model = {"name": name_token.value}
        
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after model name")
        
        # Parse model properties
        self._parse_properties(model)
        
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after model properties")
        
        # Add model to AST
        self.ast["models"].append(model)
        
    def _parse_tool(self):
        """Parse tool definition"""
        # Expect 'tool' identifier
        self._consume(TokenType.IDENTIFIER, "Expected 'tool' keyword")
        
        # Expect tool name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected tool name")
        tool = {"name": name_token.value}
        
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after tool name")
        
        # Parse tool properties
        self._parse_properties(tool)
        
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after tool properties")
        
        # Add tool to AST
        self.ast["tools"].append(tool)
        
    def _parse_magnetize(self):
        """Parse magnetic field configuration"""
        # Expect 'magnetize' identifier
        self._consume(TokenType.IDENTIFIER, "Expected 'magnetize' keyword")
        
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'magnetize'")
        
        # Parse teams and flow section
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.IDENTIFIER):
                if self._peek().value == "flow":
                    self._parse_flow()
                else:
                    self._parse_team()
            elif self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
            else:
                token = self._peek()
                raise SyntaxError(f"Unexpected token at line {token.line}: {token.type}")
            
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after magnetize block")
        
    def _parse_team(self):
        """Parse team definition within magnetize block"""
        # Expect team name
        name_token = self._consume(TokenType.IDENTIFIER, "Expected team name")
        team = {"name": name_token.value}
        
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after team name")
        
        # Parse team properties
        self._parse_properties(team)
        
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after team properties")
        
        # Add team to AST
        self.ast["teams"].append(team)
        
    def _parse_flow(self):
        """Parse flow section within magnetize block"""
        # Expect 'flow' identifier
        self._consume(TokenType.IDENTIFIER, "Expected 'flow' keyword")
        
        # Expect opening brace
        self._consume(TokenType.LBRACE, "Expected '{' after 'flow'")
        
        # Parse flow definitions
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
                continue
                
            # Expect source team identifier
            source_token = self._consume(TokenType.IDENTIFIER, "Expected source team identifier")
            source_team = source_token.value
            
            # Expect flow operator
            flow_type = None
            if self._check(TokenType.ARROW):
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
                    raise SyntaxError(f"Unknown flow operator at line {arrow_token.line}: {arrow_token.value}")
            else:
                token = self._peek()
                raise SyntaxError(f"Expected flow operator at line {token.line}, got {token.type}")
                
            # Expect target team identifier
            target_token = self._consume(TokenType.IDENTIFIER, "Expected target team identifier")
            target_team = target_token.value
            
            # Create flow definition
            flow = {
                "source": source_team,
                "target": target_team,
                "type": flow_type
            }
            
            # Add flow to AST
            self.ast["flows"].append(flow)
            
            # Check for semicolon or comment
            if self._check(TokenType.SEMICOLON):
                self._advance()
            elif self._check(TokenType.COMMENT):
                self._advance()
                
        # Expect closing brace
        self._consume(TokenType.RBRACE, "Expected '}' after flow definitions")
        
    def _parse_properties(self, target: Dict[str, Any]):
        """Parse properties into the target dictionary"""
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.COMMENT):
                # Skip comments
                self._advance()
                continue
                
            # Expect property name
            name_token = self._consume(TokenType.IDENTIFIER, "Expected property name")
            property_name = name_token.value
            
            # Check for nested object
            if self._check(TokenType.LBRACE):
                # Parse nested object
                self._advance()  # Consume '{'
                
                if property_name not in target:
                    target[property_name] = {}
                    
                self._parse_properties(target[property_name])
                
                self._consume(TokenType.RBRACE, f"Expected '}}' after {property_name} properties")
            else:
                # Expect equals sign
                self._consume(TokenType.EQUALS, f"Expected '=' after {property_name}")
                
                # Parse property value
                value = self._parse_value()
                target[property_name] = value
                
    def _parse_value(self) -> Any:
        """Parse a value (string, number, boolean, identifier, or array)"""
        token = self._advance()
        
        if token.type == TokenType.STRING:
            return token.value
        elif token.type == TokenType.NUMBER:
            # Convert to float or int
            if '.' in token.value:
                return float(token.value)
            else:
                return int(token.value)
        elif token.type == TokenType.BOOLEAN:
            return token.value.lower() == 'true'
        elif token.type == TokenType.IDENTIFIER:
            return token.value
        elif token.type == TokenType.LBRACKET:
            # Parse array
            array = []
            
            # Empty array
            if self._check(TokenType.RBRACKET):
                self._advance()
                return array
                
            # Parse array elements
            while True:
                value = self._parse_value()
                array.append(value)
                
                if self._check(TokenType.RBRACKET):
                    break
                    
                self._consume(TokenType.COMMA, "Expected ',' between array elements")
                
            self._consume(TokenType.RBRACKET, "Expected ']' after array")
            return array
        else:
            raise SyntaxError(f"Unexpected token at line {token.line}: {token.type}")
            
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
        return self.pos >= len(self.tokens) or self.tokens[self.pos].type == TokenType.EOF
        
    def _check(self, type_: TokenType) -> bool:
        """Check if the current token is of the given type"""
        if self._is_at_end():
            return False
        return self.tokens[self.pos].type == type_
        
    def _consume(self, type_: TokenType, error_message: str) -> Token:
        """Consume a token of the expected type or raise an error"""
        if self._check(type_):
            return self._advance()
        
        current = self._peek()
        raise SyntaxError(f"{error_message} at line {current.line}, got {current.type}")

class GlueDSL:
    """Main interface for GLUE DSL"""
    
    @staticmethod
    def parse_file(filename: str) -> Dict[str, Any]:
        """Parse a .glue file"""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filename}")
            
        if path.suffix != '.glue':
            raise ValueError("File must have .glue extension")
            
        with path.open('r') as f:
            content = f.read()
            
        return GlueDSL.parse(content)
    
    @staticmethod
    def parse(content: str) -> Dict[str, Any]:
        """Parse GLUE DSL content"""
        # Tokenize
        lexer = GlueLexer(content)
        tokens = lexer.tokenize()
        
        # Parse
        parser = GlueParser(tokens)
        return parser.parse()


class GlueDSLParser:
    """Parser for the GLUE Domain Specific Language.
    
    This class provides a high-level interface for parsing GLUE DSL files
    and validating their contents against the expected schema.
    """
    
    def __init__(self, schema_version: str = "1.0"):
        """Initialize a new GlueDSLParser.
        
        Args:
            schema_version: Version of the GLUE schema to use for validation
        """
        self.schema_version = schema_version
        self.logger = logging.getLogger("glue.dsl.parser")
    
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
        return GlueDSL.parse_file(file_path)
    
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
        return GlueDSL.parse(content)
    
    def validate(self, config: Dict[str, Any]) -> List[str]:
        """Validate a parsed GLUE configuration.
        
        Args:
            config: Parsed GLUE configuration
            
        Returns:
            List of validation errors, empty if valid
        """
        self.logger.info("Validating GLUE configuration")
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
