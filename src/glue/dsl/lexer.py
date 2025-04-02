"""GLUE DSL Lexer

This module provides the lexer for the GLUE DSL, which tokenizes the input
text into a stream of tokens for the parser.
"""

import re
from typing import List, Tuple, Dict, Any
import logging

from glue.dsl.parser import TokenType, Token


class GlueLexer:
    """Lexer for the GLUE DSL.
    
    The lexer tokenizes the input text into a stream of tokens for the parser.
    """
    
    # Token types
    TOKEN_TYPES = {
        'KEYWORD': r'(app|team|tool|tools|model|config|use|with|apply|glue|magnetize)',
        'IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',
        'STRING': r'"([^"\\]|\\.)*"',
        'NUMBER': r'\d+(\.\d+)?',
        'LBRACE': r'\{',
        'RBRACE': r'\}',
        'LPAREN': r'\(',
        'RPAREN': r'\)',
        'LBRACKET': r'\[',
        'RBRACKET': r'\]',
        'COLON': r':',
        'COMMA': r',',
        'EQUALS': r'=',
        'COMMENT': r'//.*',
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
        self.context_stack = []  # Stack to track lexical context
        
        # Skip newlines at the beginning of the file
        while self.pos < len(self.source) and self.source[self.pos] == '\n':
            self.pos += 1
            self.line += 1
        
        while self.pos < len(self.source):
            # Check for "apply glue" as a special case
            if self.pos + 10 <= len(self.source) and self.source[self.pos:self.pos+10].lower() == "apply glue":
                self.tokens.append(Token(TokenType.APPLY_GLUE, "apply glue", self.line))
                self.column += 10
                self.pos += 10
                continue
                
            self._tokenize_next()
            
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', self.line))
        return self.tokens
    
    def _tokenize_next(self) -> None:
        """Tokenize the next token in the source text."""
        # Try to match each pattern
        for token_type, pattern in self.patterns:
            match = pattern.match(self.source[self.pos:])
            if match:
                value = match.group(0)
                
                # Track context for braces
                if token_type == 'LBRACE':
                    # If we just saw 'config', we're entering a config block
                    if len(self.tokens) > 0 and self.tokens[-1].value == 'config':
                        self.context_stack.append('config')
                    else:
                        self.context_stack.append('block')
                elif token_type == 'RBRACE' and self.context_stack:
                    self.context_stack.pop()
                
                # Skip whitespace and comments
                if token_type not in ['WHITESPACE', 'COMMENT', 'NEWLINE']:
                    if token_type == 'KEYWORD':
                        # Check if we're in a config block
                        in_config = 'config' in self.context_stack
                        
                        # Only treat specific words as keywords at the top level
                        if value in ["app", "model", "tool", "apply", "glue", "magnetize"] and not in_config:
                            self.tokens.append(Token(TokenType.KEYWORD, value, self.line))
                        else:
                            # Treat everything else as an identifier
                            self.tokens.append(Token(TokenType.IDENTIFIER, value, self.line))
                    else:
                        # Map string token type to TokenType enum
                        enum_token_type = getattr(TokenType, token_type, TokenType.IDENTIFIER)
                        self.tokens.append(Token(enum_token_type, value, self.line))
                
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
