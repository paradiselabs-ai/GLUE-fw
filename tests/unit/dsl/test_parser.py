"""
Unit tests for the GLUE DSL Parser.
"""
import pytest
import sys
import logging
from pathlib import Path

# Directly import from source files 
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from glue.dsl.parser import TokenType, Token, GlueDSLParser
from glue.dsl.lexer import GlueLexer

class TestParserBasics:
    def test_parser_initialization(self):
        """Test that the parser initializes properly."""
        parser = GlueDSLParser()
        assert parser.schema_version == "1.0"
        assert isinstance(parser.logger, logging.Logger)

    def test_empty_tokens(self):
        """Test parser with empty tokens."""
        parser = GlueDSLParser()
        tokens = [Token(TokenType.EOF, "", 1)]
        ast = parser.parse(tokens)
        assert ast is not None
        assert len(ast.get("app", {})) == 0
        assert len(ast.get("models", {})) == 0
        assert len(ast.get("tools", {})) == 0
        assert len(ast.get("magnetize", {})) == 0

class TestParserKeywordsInConfig:
    def test_keywords_in_config_block(self):
        """Test that keywords can be used as property names in config blocks."""
        source = """
        model test_model {
            config {
                model = "gpt-4"
                tool = "web_search"
                app = "test"
            }
        }
        """
        lexer = GlueLexer()
        tokens = lexer.tokenize(source)
        parser = GlueDSLParser()
        ast = parser.parse(tokens)
        
        # Check that the config block was parsed correctly
        assert "models" in ast
        assert "test_model" in ast["models"]
        assert "config" in ast["models"]["test_model"]
        assert ast["models"]["test_model"]["config"]["model"] == "gpt-4"
        assert ast["models"]["test_model"]["config"]["tool"] == "web_search"
        assert ast["models"]["test_model"]["config"]["app"] == "test"

    def test_keywords_in_array(self):
        """Test that keywords can be used as values in arrays."""
        source = """
        model test_model {
            adhesives = [glue, model, tool]
        }
        """
        lexer = GlueLexer()
        tokens = lexer.tokenize(source)
        parser = GlueDSLParser()
        ast = parser.parse(tokens)
        
        # Check that the array was parsed correctly
        assert "models" in ast
        assert "test_model" in ast["models"]
        assert "adhesives" in ast["models"]["test_model"]
        assert ast["models"]["test_model"]["adhesives"] == ["glue", "model", "tool"]

class TestParserMagnetize:
    def test_magnetize_block(self):
        """Test that magnetize blocks are parsed correctly."""
        source = """
        model test_model {
            provider = "openai"
        }
        
        tool web_search {
            provider = "serp"
        }
        
        magnetize {
            research_team {
                lead = test_model
                tools = [web_search]
            }
        }
        """
        lexer = GlueLexer()
        tokens = lexer.tokenize(source)
        parser = GlueDSLParser()
        ast = parser.parse(tokens)
        
        # Check that the magnetize block was parsed correctly
        assert "magnetize" in ast
        assert "research_team" in ast["magnetize"]
        assert "lead" in ast["magnetize"]["research_team"]
        assert ast["magnetize"]["research_team"]["lead"] == "test_model"
        assert "tools" in ast["magnetize"]["research_team"]
        assert ast["magnetize"]["research_team"]["tools"] == ["web_search"]

class TestRealWorldExamples:
    def test_gemini_example(self):
        """Test parsing the Gemini example file."""
        source = """
        glue app {
            name = "Gemini Test"
            config {
                development = true
            }
        }
        
        model gemini_model {
            provider = gemini
            role = "Answer questions and provide assistance"
            adhesives = [glue, velcro]
            config {
                model = "gemini-1.5-pro"
                temperature = 0.7
                max_tokens = 1024
            }
        }
        
        tool web_search {
            provider = serp
        }
        
        magnetize {
            research_team {
                lead = gemini_model
                tools = [web_search]
            }
        }
        
        apply glue
        """
        lexer = GlueLexer()
        tokens = lexer.tokenize(source)
        parser = GlueDSLParser()
        ast = parser.parse(tokens)
        
        # Check that all sections were parsed correctly
        assert "app" in ast
        assert ast["app"]["name"] == "Gemini Test"
        assert ast["app"]["config"]["development"] == True
        
        assert "models" in ast
        assert "gemini_model" in ast["models"]
        assert ast["models"]["gemini_model"]["provider"] == "gemini"
        assert ast["models"]["gemini_model"]["role"] == "Answer questions and provide assistance"
        assert ast["models"]["gemini_model"]["adhesives"] == ["glue", "velcro"]
        assert ast["models"]["gemini_model"]["config"]["model"] == "gemini-1.5-pro"
        assert ast["models"]["gemini_model"]["config"]["temperature"] == 0.7
        assert ast["models"]["gemini_model"]["config"]["max_tokens"] == 1024
        
        assert "tools" in ast
        assert "web_search" in ast["tools"]
        assert ast["tools"]["web_search"]["provider"] == "serp"
        
        assert "magnetize" in ast
        assert "research_team" in ast["magnetize"]
        assert ast["magnetize"]["research_team"]["lead"] == "gemini_model"
        assert ast["magnetize"]["research_team"]["tools"] == ["web_search"]
