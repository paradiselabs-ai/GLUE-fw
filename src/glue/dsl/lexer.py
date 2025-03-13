"""GLUE DSL Lexer

This module provides the lexer for the GLUE DSL, which tokenizes the input
text into a stream of tokens for the parser.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class Token:
    """Token in the GLUE DSL."""
    
    def __init__(self, type_: str, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
        
    def __repr__(self):
        return f"Token({self.type}, '{self.value}', line={self.line}, col={self.column})"


class GlueLexer:
    """Lexer for the GLUE DSL.
    
    The lexer tokenizes the input text into a stream of tokens for the parser.
    """
    
    # Token types
    TOKEN_TYPES = {
        'KEYWORD': r'(app|team|tool|model|config|use|with)',
        'IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',
        'STRING': r'"([^"\\]|\\.)*"',
        'NUMBER': r'\d+(\.\d+)?',
        'LBRACE': r'\{',
        'RBRACE': r'\}',
        'LPAREN': r'\(',
        'RPAREN': r'\)',
        'COLON': r':',
        'COMMA': r',',
        'EQUALS': r'=',
        'COMMENT': r'#.*',
        'WHITESPACE': r'[ \t]+',
        'NEWLINE': r'\n',
    }
    
    def __init__(self):
        """Initialize a new GlueLexer."""
        self.tokens: List[Token] = []
        self.source: str = ""
        self.pos: int = 0
        self.line: int = 1
        self.column: int = 1
        
        # Compile regex patterns
        self.patterns = []
        for token_type, pattern in self.TOKEN_TYPES.items():
            self.patterns.append((token_type, re.compile(pattern)))
    
    def tokenize(self, source: str) -> List[Token]:
        """Tokenize the input text.
        
        Args:
            source: The source text to tokenize
            
        Returns:
            List of tokens
        """
        self.source = source
        self.tokens = []
        self.pos = 0
        self.line = 1
        self.column = 1
        
        while self.pos < len(self.source):
            self._tokenize_next()
            
        # Add EOF token
        self.tokens.append(Token('EOF', '', self.line, self.column))
        return self.tokens
    
    def _tokenize_next(self) -> None:
        """Tokenize the next token in the source text."""
        # Try to match each pattern
        for token_type, pattern in self.patterns:
            match = pattern.match(self.source[self.pos:])
            if match:
                value = match.group(0)
                
                # Skip whitespace and comments
                if token_type not in ['WHITESPACE', 'COMMENT']:
                    self.tokens.append(Token(token_type, value, self.line, self.column))
                
                # Update position and line/column counters
                if token_type == 'NEWLINE':
                    self.line += 1
                    self.column = 1
                else:
                    self.column += len(value)
                
                self.pos += len(value)
                return
        
        # If we get here, no pattern matched
        raise SyntaxError(f"Invalid syntax at line {self.line}, column {self.column}: '{self.source[self.pos]}'")
