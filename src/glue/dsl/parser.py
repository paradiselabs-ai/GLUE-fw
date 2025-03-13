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
        
        while self.pos < len(self.source) and self.source[self.pos] != delimiter:
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
            if self._match(TokenType.IDENTIFIER, "glue"):
                self._parse_app()
            elif self._match(TokenType.IDENTIFIER, "model"):
                self._parse_model()
            elif self._match(TokenType.IDENTIFIER, "tool"):
                self._parse_tool()
            elif self._match(TokenType.IDENTIFIER, "magnetize"):
                self._parse_magnetize()
            else:
                self._advance()
                
        return self.ast
        
    def _parse_app(self):
        """Parse app configuration"""
        # Expect: glue app { ... }
        if not self._match(TokenType.IDENTIFIER, "app"):
            self._error("Expected 'app' after 'glue'")
            
        if not self._match(TokenType.LBRACE):
            self._error("Expected '{' after 'app'")
            
        app_config = self._parse_block()
        self.ast["app"] = app_config
        
    def _parse_model(self):
        """Parse model definition"""
        name = self._consume(TokenType.IDENTIFIER, "Expected model name")
        if not self._match(TokenType.LBRACE):
            self._error("Expected '{' after model name")
            
        config = self._parse_block()
        model = {
            "name": name.value,
            "config": ModelConfig(
                provider=config.get("provider", ""),
                model_id=config.get("model", ""),
                temperature=float(config.get("temperature", 0.7)),
                max_tokens=int(config.get("max_tokens", 2048))
            )
        }
        self.ast["models"].append(model)
        
    def _parse_tool(self):
        """Parse tool definition"""
        name = self._consume(TokenType.IDENTIFIER, "Expected tool name")
        if not self._match(TokenType.LBRACE):
            self._error("Expected '{' after tool name")
            
        config = self._parse_block()
        tool = {
            "name": name.value,
            "config": ToolConfig(
                name=name.value,
                description=config.get("description", ""),
                provider=config.get("provider"),
                config=config.get("config", {})
            )
        }
        self.ast["tools"].append(tool)
        
    def _parse_magnetize(self):
        """Parse magnetic field configuration"""
        if not self._match(TokenType.LBRACE):
            self._error("Expected '{' after 'magnetize'")
            
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            name = self._consume(TokenType.IDENTIFIER, "Expected team name")
            
            if self._match(TokenType.LBRACE):
                # Team definition
                team_config = self._parse_block()
                team = {
                    "name": name.value,
                    "config": TeamConfig(
                        name=name.value,
                        lead=team_config.get("lead", ""),
                        members=team_config.get("members", []),
                        tools=team_config.get("tools", [])
                    )
                }
                self.ast["teams"].append(team)
            elif self._match(TokenType.ARROW):
                # Magnetic flow definition
                operator = self._previous().value
                target = self._consume(TokenType.IDENTIFIER, "Expected target team")
                
                flow = {
                    "source": name.value,
                    "target": target.value,
                    "type": FlowType(operator)
                }
                self.ast["flows"].append(flow)
                
        self._consume(TokenType.RBRACE, "Expected '}' after magnetize block")

    # ... [Helper methods like _parse_block, _match, _advance, etc. continue] ...

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
