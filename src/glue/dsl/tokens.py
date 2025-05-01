"""GLUE DSL Token Types and Definitions

This module contains the token types and token class used by both the lexer and parser.
"""

from enum import Enum
from dataclasses import dataclass


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
    ARROW = "ARROW"  # For magnetic operators
    SEMICOLON = "SEMICOLON"
    COMMENT = "COMMENT"
    EOF = "EOF"
    KEYWORD = "KEYWORD"
    APPLY_GLUE = "APPLY_GLUE"


@dataclass
class Token:
    """Token representation"""

    type: TokenType
    value: str
    line: int
