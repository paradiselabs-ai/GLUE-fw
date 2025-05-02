"""
Isolated unit tests for the GLUE DSL Parser.
These tests focus on the parser's ability to convert tokens into an AST.
"""
import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any

# Directly import from source files 
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from glue.dsl.parser import TokenType, Token, GlueParser

# ==================== Mock Classes ====================
class MockLexer:
    """Mock lexer for testing the parser in isolation."""
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        
    def tokenize(self) -> List[Token]:
        return self.tokens

# ==================== Helper Functions ====================
def create_token(type_: TokenType, value: str, line: int = 1) -> Token:
    """Create a token for testing."""
    return Token(type_, value, line)

def create_tokens_for_app() -> List[Token]:
    """Create tokens for a basic app definition."""
    return [
        create_token(TokenType.KEYWORD, "glue"),
        create_token(TokenType.KEYWORD, "app"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "name"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.STRING, "\"Test App\""),
        create_token(TokenType.IDENTIFIER, "config"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "development"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.BOOLEAN, "true"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.EOF, "")
    ]

def create_tokens_for_model() -> List[Token]:
    """Create tokens for a basic model definition."""
    return [
        create_token(TokenType.KEYWORD, "model"),
        create_token(TokenType.IDENTIFIER, "test_model"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "provider"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.KEYWORD, "openai"),
        create_token(TokenType.IDENTIFIER, "config"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "model"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.STRING, "\"gpt-4\""),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.EOF, "")
    ]

def create_tokens_for_tool() -> List[Token]:
    """Create tokens for a basic tool definition."""
    return [
        create_token(TokenType.KEYWORD, "tool"),
        create_token(TokenType.IDENTIFIER, "web_search"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "provider"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.KEYWORD, "serp"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.EOF, "")
    ]

def create_tokens_for_magnetize() -> List[Token]:
    """Create tokens for a basic magnetize block."""
    return [
        create_token(TokenType.KEYWORD, "magnetize"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "research_team"),
        create_token(TokenType.LBRACE, "{"),
        create_token(TokenType.IDENTIFIER, "lead"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.IDENTIFIER, "test_model"),
        create_token(TokenType.IDENTIFIER, "tools"),
        create_token(TokenType.EQUALS, "="),
        create_token(TokenType.LBRACKET, "["),
        create_token(TokenType.IDENTIFIER, "web_search"),
        create_token(TokenType.RBRACKET, "]"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.RBRACE, "}"),
        create_token(TokenType.EOF, "")
    ]

def create_tokens_for_complete_app() -> List[Token]:
    """Create tokens for a complete app with all sections."""
    app_tokens = create_tokens_for_app()
    model_tokens = create_tokens_for_model()
    tool_tokens = create_tokens_for_tool()
    magnetize_tokens = create_tokens_for_magnetize()
    
    # Remove EOF tokens except from the last section
    app_tokens.pop()  # Remove EOF
    model_tokens.pop()  # Remove EOF
    tool_tokens.pop()  # Remove EOF
    
    # Combine all tokens
    return app_tokens + model_tokens + tool_tokens + magnetize_tokens

def create_parser_with_empty_ast() -> GlueParser:
    """Create a parser with an empty AST."""
    parser = GlueParser()
    parser.ast = {
        "app": {},
        "teams": [],
        "models": {},
        "tools": {},
        "flows": [],
        "magnetize": {},
        "apply": None
    }
    return parser

# ==================== Parser Initialization Tests ====================
class TestParserInitialization:
    def test_parser_initialization(self):
        """Test that the parser initializes correctly."""
        parser = create_parser_with_empty_ast()
        
        assert parser.ast == {
            "app": {},
            "teams": [],
            "models": {},
            "tools": {},
            "flows": [],
            "magnetize": {},
            "apply": None
        }

# ==================== App Parsing Tests ====================
class TestAppParsing:
    def test_parse_app(self):
        """Test parsing a basic app definition."""
        tokens = create_tokens_for_app()
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that the app was parsed correctly
        assert "app" in parser.ast
        assert parser.ast["app"]["name"] == "Test App"
        assert "config" in parser.ast["app"]
        assert parser.ast["app"]["config"]["development"] is True
        
    def test_parse_app_missing_name(self):
        """Test parsing an app with missing name."""
        tokens = [
            create_token(TokenType.KEYWORD, "glue"),
            create_token(TokenType.KEYWORD, "app"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "config"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "development"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.BOOLEAN, "true"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should not raise an error, but app should not have a name
        parser.parse()
        assert "app" in parser.ast
        assert "name" not in parser.ast["app"]
        
    def test_parse_app_invalid_structure(self):
        """Test parsing an app with invalid structure."""
        tokens = [
            create_token(TokenType.KEYWORD, "glue"),
            create_token(TokenType.KEYWORD, "app"),
            # Missing opening brace
            create_token(TokenType.IDENTIFIER, "name"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.STRING, "Test App"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should raise a syntax error
        with pytest.raises(SyntaxError):
            parser.parse()

# ==================== Model Parsing Tests ====================
class TestModelParsing:
    def test_parse_model(self):
        """Test parsing a model definition."""
        tokens = create_tokens_for_model()
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that the model was parsed correctly
        assert "test_model" in parser.ast["models"]
        model = parser.ast["models"]["test_model"]
        assert model["provider"] == "openai"
        assert "config" in model
        assert model["config"]["model"] == "gpt-4"
        
    def test_parse_model_missing_required_fields(self):
        """Test parsing a model with missing required fields."""
        tokens = [
            create_token(TokenType.KEYWORD, "model"),
            create_token(TokenType.IDENTIFIER, "test_model"),
            create_token(TokenType.LBRACE, "{"),
            # Missing provider and role
            create_token(TokenType.IDENTIFIER, "config"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "model"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.STRING, "test/model"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should parse but model should be incomplete
        parser.parse()
        assert "test_model" in parser.ast["models"]
        model = parser.ast["models"]["test_model"]
        assert "provider" not in model

# ==================== Tool Parsing Tests ====================
class TestToolParsing:
    def test_parse_tool(self):
        """Test parsing a tool definition."""
        tokens = create_tokens_for_tool()
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that the tool was parsed correctly
        assert "web_search" in parser.ast["tools"]
        tool = parser.ast["tools"]["web_search"]
        assert tool["provider"] == "serp"
        
    def test_parse_tool_empty(self):
        """Test parsing an empty tool definition."""
        tokens = [
            create_token(TokenType.KEYWORD, "tool"),
            create_token(TokenType.IDENTIFIER, "empty_tool"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that the tool was parsed correctly
        assert "empty_tool" in parser.ast["tools"]
        tool = parser.ast["tools"]["empty_tool"]
        assert len(tool) == 0

# ==================== Magnetize Parsing Tests ====================
class TestMagnetizeParsing:
    def test_parse_magnetize(self):
        """Test parsing a magnetize block."""
        tokens = create_tokens_for_magnetize()
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that the team was parsed correctly
        assert "research_team" in parser.ast["magnetize"]
        team = parser.ast["magnetize"]["research_team"]
        assert team["lead"] == "test_model"
        assert "tools" in team
        assert team["tools"] == ["web_search"]
        
    def test_parse_magnetize_multiple_teams(self):
        """Test parsing a magnetize block with multiple teams."""
        tokens = [
            create_token(TokenType.KEYWORD, "magnetize"),
            create_token(TokenType.LBRACE, "{"),
            # First team
            create_token(TokenType.IDENTIFIER, "team1"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "lead"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.IDENTIFIER, "model1"),
            create_token(TokenType.RBRACE, "}"),
            # Second team
            create_token(TokenType.IDENTIFIER, "team2"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "lead"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.IDENTIFIER, "model2"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that both teams were parsed correctly
        assert "team1" in parser.ast["magnetize"]
        assert "team2" in parser.ast["magnetize"]
        assert parser.ast["magnetize"]["team1"]["lead"] == "model1"
        assert parser.ast["magnetize"]["team2"]["lead"] == "model2"

# ==================== Complete App Parsing Tests ====================
class TestCompleteAppParsing:
    def test_parse_complete_app(self):
        """Test parsing a complete app with all components."""
        tokens = create_tokens_for_complete_app()
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that all components were parsed correctly
        assert "app" in parser.ast
        assert parser.ast["app"]["name"] == "Test App"
        
        assert "test_model" in parser.ast["models"]
        assert parser.ast["models"]["test_model"]["provider"] == "openai"
        
        assert "web_search" in parser.ast["tools"]
        assert parser.ast["tools"]["web_search"]["provider"] == "serp"
        
        assert "research_team" in parser.ast["magnetize"]
        assert parser.ast["magnetize"]["research_team"]["lead"] == "test_model"
        assert parser.ast["magnetize"]["research_team"]["tools"] == ["web_search"]
        
    def test_parse_with_comments(self):
        """Test parsing with comments interspersed."""
        # Start with basic app tokens
        tokens = create_tokens_for_app()[:-1]  # Remove EOF
        
        # Add comments
        comment_tokens = [
            create_token(TokenType.COMMENT, "// This is a model definition"),
            create_token(TokenType.KEYWORD, "model"),
            create_token(TokenType.IDENTIFIER, "test_model"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.COMMENT, "// Provider configuration"),
            create_token(TokenType.IDENTIFIER, "provider"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.KEYWORD, "openai"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        
        tokens.extend(comment_tokens)
        parser = GlueParser(tokens)
        
        parser.parse()
        
        # Check that comments were ignored and components were parsed correctly
        assert "app" in parser.ast
        assert "test_model" in parser.ast["models"]
        assert parser.ast["models"]["test_model"]["provider"] == "openai"

# ==================== Error Handling Tests ====================
class TestErrorHandling:
    def test_unexpected_token(self):
        """Test handling of unexpected tokens."""
        tokens = [
            create_token(TokenType.IDENTIFIER, "unknown"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should raise a syntax error for unknown top-level identifier
        with pytest.raises(SyntaxError):
            parser.parse()
            
    def test_missing_closing_brace(self):
        """Test handling of missing closing braces."""
        tokens = [
            create_token(TokenType.KEYWORD, "glue"),
            create_token(TokenType.KEYWORD, "app"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "name"),
            create_token(TokenType.EQUALS, "="),
            create_token(TokenType.STRING, "Test App"),
            # Missing closing brace
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should raise a syntax error
        with pytest.raises(SyntaxError):
            parser.parse()
            
    def test_missing_equals(self):
        """Test handling of missing equals sign."""
        tokens = [
            create_token(TokenType.KEYWORD, "glue"),
            create_token(TokenType.KEYWORD, "app"),
            create_token(TokenType.LBRACE, "{"),
            create_token(TokenType.IDENTIFIER, "name"),
            # Missing equals sign
            create_token(TokenType.STRING, "Test App"),
            create_token(TokenType.RBRACE, "}"),
            create_token(TokenType.EOF, "")
        ]
        parser = GlueParser(tokens)
        
        # Should raise a syntax error
        with pytest.raises(SyntaxError):
            parser.parse()

# ==================== Integration with Lexer Tests ====================
class TestLexerIntegration:
    def test_lexer_parser_integration(self):
        """Test integration between lexer and parser."""
        # This test would normally use the real lexer, but for isolation we use our mock
        source = """
        glue app {
            name = "Test App"
            config {
                development = true
            }
        }
        """
        
        # Use our token generator instead of a real lexer
        tokens = create_tokens_for_app()
        mock_lexer = MockLexer(tokens)
        
        # Parse the tokens
        parser = GlueParser(mock_lexer.tokenize())
        parser.parse()
        
        # Check the result
        assert parser.ast["app"]["name"] == "Test App"
        assert parser.ast["app"]["config"]["development"] is True
