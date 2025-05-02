"""
Unit tests for the GLUE DSL Lexer.
"""
import pytest
import sys
from pathlib import Path

# Directly import from source files 
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from glue.dsl.parser import TokenType, Token, GlueLexer

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
    def test_simple_app_definition(self, simple_glue_content):
        """Test tokenization of a simple GLUE app definition."""
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

    def test_real_example(self, example_glue_content):
        """Test tokenization of the real example file."""
        lexer = GlueLexer(example_glue_content)
        tokens = lexer.tokenize()
        
        # Just check that we get tokens and EOF is the last one
        assert len(tokens) > 0
        assert tokens[-1].type == TokenType.EOF
        
        # Basic validation of token counts
        identifier_count = sum(1 for t in tokens if t.type == TokenType.IDENTIFIER)
        string_count = sum(1 for t in tokens if t.type == TokenType.STRING)
        lbrace_count = sum(1 for t in tokens if t.type == TokenType.LBRACE)
        rbrace_count = sum(1 for t in tokens if t.type == TokenType.RBRACE)
        
        # Check that braces are balanced
        assert lbrace_count == rbrace_count
        # Check that we have several identifiers and strings
        assert identifier_count > 10
        assert string_count > 3
