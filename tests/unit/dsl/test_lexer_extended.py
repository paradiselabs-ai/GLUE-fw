"""
Extended unit tests for the GLUE DSL Lexer.
These tests focus on edge cases and more complex scenarios.
"""
import pytest
import sys
from pathlib import Path

# Directly import from source files 
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from glue.dsl.parser import TokenType, Token, GlueLexer

# ==================== Nested Structure Tests ====================
class TestNestedStructures:
    def test_deeply_nested_braces(self):
        """Test tokenization of deeply nested brace structures."""
        source = """
        glue app {
            config {
                advanced {
                    settings {
                        debug {
                            level = 3
                        }
                    }
                }
            }
        }
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Count opening and closing braces
        lbrace_count = sum(1 for t in tokens if t.type == TokenType.LBRACE)
        rbrace_count = sum(1 for t in tokens if t.type == TokenType.RBRACE)
        
        assert lbrace_count == 5, f"Expected 5 opening braces, got {lbrace_count}"
        assert rbrace_count == 5, f"Expected 5 closing braces, got {rbrace_count}"
        
    def test_nested_arrays(self):
        """Test tokenization of nested arrays."""
        source = """
        tools = [
            [search, code],
            [file, network]
        ]
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Count brackets and identifiers
        lbracket_count = sum(1 for t in tokens if t.type == TokenType.LBRACKET)
        rbracket_count = sum(1 for t in tokens if t.type == TokenType.RBRACKET)
        identifier_count = sum(1 for t in tokens if t.type == TokenType.IDENTIFIER)
        
        assert lbracket_count == 3, f"Expected 3 opening brackets, got {lbracket_count}"
        assert rbracket_count == 3, f"Expected 3 closing brackets, got {rbracket_count}"
        assert identifier_count == 5, f"Expected 5 identifiers, got {identifier_count}"  # tools + 4 tool names

# ==================== Whitespace Handling Tests ====================
class TestWhitespaceHandling:
    def test_various_whitespace(self):
        """Test handling of various whitespace patterns."""
        source = """
        model   test_model    {
        provider=openrouter
            role    =    "Test model"
        }
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Filter out EOF token
        tokens = [t for t in tokens if t.type != TokenType.EOF]
        
        # Check that whitespace is properly ignored but tokens are correctly identified
        assert len(tokens) == 10, f"Expected 10 tokens, got {len(tokens)}"
        assert tokens[0].type == TokenType.IDENTIFIER and tokens[0].value == "model"
        assert tokens[1].type == TokenType.IDENTIFIER and tokens[1].value == "test_model"
        
    def test_tabs_and_spaces(self):
        """Test handling of tabs and spaces."""
        source = "model\ttest_model\t{\n\tprovider\t=\topenrouter\n}"
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Filter out EOF token
        tokens = [t for t in tokens if t.type != TokenType.EOF]
        
        assert len(tokens) == 7, f"Expected 7 tokens, got {len(tokens)}"
        assert tokens[0].type == TokenType.IDENTIFIER and tokens[0].value == "model"
        assert tokens[1].type == TokenType.IDENTIFIER and tokens[1].value == "test_model"

# ==================== Line Number Tracking Tests ====================
class TestLineNumberTracking:
    def test_multiline_input(self):
        """Test line number tracking across multiline input."""
        source = """
        // Line 1
        model test_model { // Line 3
            // Line 4
            provider = openrouter // Line 5
            // Line 6
            role = "Test model" // Line 7
        } // Line 8
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Find tokens and check their line numbers
        model_token = next(t for t in tokens if t.type == TokenType.IDENTIFIER and t.value == "model")
        provider_token = next(t for t in tokens if t.type == TokenType.IDENTIFIER and t.value == "provider")
        role_token = next(t for t in tokens if t.type == TokenType.IDENTIFIER and t.value == "role")
        closing_brace = next(t for t in tokens if t.type == TokenType.RBRACE)
        
        assert model_token.line == 3, f"Expected 'model' on line 3, got line {model_token.line}"
        assert provider_token.line == 5, f"Expected 'provider' on line 5, got line {provider_token.line}"
        assert role_token.line == 7, f"Expected 'role' on line 7, got line {role_token.line}"
        assert closing_brace.line == 8, f"Expected closing brace on line 8, got line {closing_brace.line}"
        
    def test_line_tracking_with_strings(self):
        """Test line number tracking with multiline strings."""
        source = """
        model test_model {
            description = "This is a
            multiline
            string"
            provider = openrouter
        }
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Find tokens and check their line numbers
        string_token = next(t for t in tokens if t.type == TokenType.STRING)
        provider_token = next(t for t in tokens if t.type == TokenType.IDENTIFIER and t.value == "provider")
        
        # Check that the provider token is on line 6 (after the multiline string)
        assert provider_token.line == 6, f"Expected 'provider' on line 6, got line {provider_token.line}"
        
        # Check that the string value contains newlines
        assert "\n" in string_token.value, "Expected newlines in string value"

# ==================== Special Character Tests ====================
class TestSpecialCharacters:
    def test_special_chars_in_strings(self):
        """Test handling of special characters in strings."""
        source = """
        model test_model {
            prompt = "Special chars: !@#$%^&*()_+-=[]{}|;:'\\",.<>/?"
        }
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Find the string token
        string_token = next(t for t in tokens if t.type == TokenType.STRING)
        
        # Check that all special characters are preserved
        expected_chars = "!@#$%^&*()_+-=[]{}|;:',.< >/?"
        for char in expected_chars:
            assert char in string_token.value, f"Expected character '{char}' in string value"
            
    def test_escaped_quotes_in_strings(self):
        """Test handling of escaped quotes in strings."""
        source = """
        model test_model {
            prompt = "String with \\"escaped quotes\\""
        }
        """
        lexer = GlueLexer(source)
        tokens = lexer.tokenize()
        
        # Find the string token
        string_token = next(t for t in tokens if t.type == TokenType.STRING)
        
        # Check that escaped quotes are preserved
        assert "\\\"" in string_token.value, "Expected escaped quotes in string value"

# ==================== Performance Tests ====================
class TestLexerPerformance:
    def test_large_input(self):
        """Test lexer performance with large input."""
        # Generate a large input with repeated patterns
        repeated_pattern = "model test_model { provider = openrouter role = \"Test model\" }\n"
        large_input = repeated_pattern * 100  # 100 repetitions
        
        lexer = GlueLexer(large_input)
        
        # Measure tokenization time
        import time
        start_time = time.time()
        tokens = lexer.tokenize()
        end_time = time.time()
        
        # Check that tokenization completes in a reasonable time (adjust as needed)
        assert end_time - start_time < 1.0, f"Tokenization took too long: {end_time - start_time:.2f} seconds"
        
        # Check that all tokens were processed
        assert len(tokens) > 500, f"Expected more than 500 tokens, got {len(tokens)}"
