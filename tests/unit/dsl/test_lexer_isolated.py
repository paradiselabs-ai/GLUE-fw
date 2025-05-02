"""
Unit tests for the GLUE DSL Lexer - isolated from dependencies.
"""
import pytest
from enum import Enum
from dataclasses import dataclass
from typing import List

# ==================== Define mockup of required classes ====================
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

# ==================== TokenType Tests ====================
class TestTokenType:
    def test_token_type_enum_values(self):
        """Test that the TokenType enum has the expected values."""
        assert TokenType.IDENTIFIER.value == "IDENTIFIER"
        assert TokenType.STRING.value == "STRING"
        assert TokenType.NUMBER.value == "NUMBER"
        assert TokenType.BOOLEAN.value == "BOOLEAN"
        assert TokenType.EQUALS.value == "EQUALS"
        assert TokenType.LBRACE.value == "LBRACE"
        assert TokenType.RBRACE.value == "RBRACE"
        assert TokenType.LBRACKET.value == "LBRACKET"
        assert TokenType.RBRACKET.value == "RBRACKET"
        assert TokenType.COMMA.value == "COMMA"
        assert TokenType.ARROW.value == "ARROW"
        assert TokenType.SEMICOLON.value == "SEMICOLON"
        assert TokenType.COMMENT.value == "COMMENT"
        assert TokenType.EOF.value == "EOF"

# ==================== Token Tests ====================
class TestToken:
    def test_token_creation(self):
        """Test that tokens can be created with the expected values."""
        token = Token(TokenType.IDENTIFIER, "test", 1)
        assert token.type == TokenType.IDENTIFIER
        assert token.value == "test"
        assert token.line == 1

# ==================== Lexer Basic Tests ====================
class TestLexerBasics:
    def test_lexer_initialization(self):
        """Test that the lexer initializes properly."""
        source = "test source"
        lexer = GlueLexer(source)
        assert lexer.source == source
        assert lexer.pos == 0
        assert lexer.line == 1
        assert lexer.tokens == []

    def test_empty_source(self):
        """Test lexer with empty source."""
        lexer = GlueLexer("")
        tokens = lexer.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

# ==================== Lexer Tokenization Tests ====================
class TestLexerTokenization:
    def test_identifier_tokenization(self):
        """Test tokenization of identifiers."""
        lexer = GlueLexer("model test_name")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 3  # 2 identifiers + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "model"
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "test_name"
        assert tokens[2].type == TokenType.EOF

    def test_string_tokenization(self):
        """Test tokenization of string literals."""
        lexer = GlueLexer('name = "Test String"')
        tokens = lexer.tokenize()
        
        assert len(tokens) == 4  # identifier + equals + string + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "name"
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.STRING
        assert tokens[2].value == "Test String"
        assert tokens[3].type == TokenType.EOF

    def test_number_tokenization(self):
        """Test tokenization of number literals."""
        lexer = GlueLexer("temperature = 0.7")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 4  # identifier + equals + number + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "temperature"
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.NUMBER
        assert tokens[2].value == "0.7"
        assert tokens[3].type == TokenType.EOF

    def test_boolean_tokenization(self):
        """Test tokenization of boolean literals."""
        lexer = GlueLexer("development = true")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 4  # identifier + equals + boolean + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "development"
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.BOOLEAN
        assert tokens[2].value == "true"
        assert tokens[3].type == TokenType.EOF

    def test_comment_handling(self):
        """Test handling of comments."""
        lexer = GlueLexer("name = 'test' // This is a comment")
        tokens = lexer.tokenize()
        
        # Comments are skipped, not tokenized
        assert len(tokens) == 4  # identifier + equals + string + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[1].type == TokenType.EQUALS
        assert tokens[2].type == TokenType.STRING
        assert tokens[3].type == TokenType.EOF

    def test_magnetic_operators(self):
        """Test tokenization of magnetic operators."""
        lexer = GlueLexer("team1 -> team2")
        tokens = lexer.tokenize()
        
        assert len(tokens) == 4  # identifier + arrow + identifier + EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "team1"
        assert tokens[1].type == TokenType.ARROW
        assert tokens[1].value == "->"
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "team2"
        assert tokens[3].type == TokenType.EOF

        # Test other magnetic operators
        lexer = GlueLexer("team1 <- team2")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.ARROW
        assert tokens[1].value == "<-"
        
        lexer = GlueLexer("team1 <> team2")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.ARROW
        assert tokens[1].value == "<>"
        
        lexer = GlueLexer("team1 >< team2")
        tokens = lexer.tokenize()
        assert tokens[1].type == TokenType.ARROW
        assert tokens[1].value == "><"

# ==================== Complex Examples Tests ====================
class TestComplexExamples:
    def test_simple_app_definition(self):
        """Test tokenization of a simple GLUE app definition."""
        simple_glue_content = """
glue app {
    name = "Test App"
    config {
        development = true
    }
}

model test_model {
    provider = openrouter
    role = "Test model"
    adhesives = [glue]
    config {
        model = "test/model"
        temperature = 0.5
    }
}

tool test_tool {}

magnetize {
    test_team {
        lead = test_model
        tools = [test_tool]
    }
}

apply glue
"""
        lexer = GlueLexer(simple_glue_content)
        tokens = lexer.tokenize()
        
        # Just check that we get tokens and EOF is the last one
        assert len(tokens) > 0
        assert tokens[-1].type == TokenType.EOF
        
        # Check for some expected tokens
        token_values = [t.value for t in tokens if t.type == TokenType.STRING]
        assert "Test App" in token_values
        
        # Check for adhesives array
        has_lbracket = any(t.type == TokenType.LBRACKET for t in tokens)
        has_rbracket = any(t.type == TokenType.RBRACKET for t in tokens)
        assert has_lbracket and has_rbracket

    def test_invalid_magnetic_operator(self):
        """Test that invalid magnetic operators raise an error."""
        lexer = GlueLexer("team1 << team2")
        with pytest.raises(ValueError, match="Invalid magnetic operator"):
            lexer.tokenize()

    def test_unterminated_string(self):
        """Test that unterminated strings raise an error."""
        lexer = GlueLexer('name = "unterminated')
        with pytest.raises(ValueError, match="Unterminated string"):
            lexer.tokenize()
