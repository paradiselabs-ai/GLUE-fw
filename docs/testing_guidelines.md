# GLUE Framework Testing Guidelines

## Overview

This document outlines the testing standards and best practices for the GLUE framework. Following these guidelines ensures consistent, high-quality tests that support the framework's reliability and maintainability.

## Test-Driven Development (TDD)

The GLUE framework follows the Test-Driven Development approach:

1. **Red**: Write failing tests that define the expected behavior
2. **Green**: Implement the minimal code to make the tests pass
3. **Refactor**: Clean up the code while keeping tests green

## Test Directory Structure

```
tests/
├── conftest.py            # Global fixtures and configuration
├── unit/                  # Unit tests
│   ├── core/              # Core component tests
│   ├── dsl/               # DSL parser tests
│   ├── tools/             # Tool tests
│   └── ...
├── integration/           # Integration tests
│   ├── dsl/               # DSL integration tests
│   ├── examples/          # Example application tests
│   └── ...
└── utils/                 # Test utilities
```

## Test Naming Conventions

- Test files: `test_<component_name>.py`
- Test classes: `Test<ComponentName><Functionality>`
- Test methods: `test_<functionality>_<scenario>`

## Test Categories

### Unit Tests

Unit tests focus on testing individual components in isolation:

- Should be fast and independent
- Use mocks for external dependencies
- Focus on a single function or class
- Cover edge cases and error conditions

### Integration Tests

Integration tests verify that components work together correctly:

- Test interactions between multiple components
- Verify end-to-end workflows
- Test with realistic inputs and configurations

## Test Coverage

- Aim for at least 90% code coverage for all components
- Use `pytest-cov` to measure coverage
- Exclude appropriate files (see `.coveragerc`)

## Writing Effective Tests

### Test Structure (AAA Pattern)

1. **Arrange**: Set up the test data and conditions
2. **Act**: Perform the action being tested
3. **Assert**: Verify the results

Example:
```python
def test_parse_simple_expression():
    # Arrange
    parser = Parser("x = 5")
    
    # Act
    result = parser.parse()
    
    # Assert
    assert result["x"] == 5
```

### Test Fixtures

Use pytest fixtures for common setup and teardown:

- Define fixtures in `conftest.py` for global use
- Use module-level fixtures for component-specific setup
- Parameterize tests to cover multiple scenarios

### Mocking

Use mocks to isolate components from their dependencies:

- Use `unittest.mock` or `pytest-mock`
- Mock external APIs and services
- Verify that mocks are called correctly

## Running Tests

### Basic Test Run

```bash
pytest
```

### With Coverage

```bash
pytest --cov=src/glue
```

### Specific Test Categories

```bash
pytest -m unit  # Run only unit tests
pytest -m integration  # Run only integration tests
```

## Continuous Integration

All tests are automatically run on GitHub Actions:

- Tests run on every push and pull request
- Coverage reports are generated and uploaded
- Linting checks ensure code quality

## Adding New Tests

When adding new functionality:

1. Start by writing tests that define the expected behavior
2. Implement the minimal code to make the tests pass
3. Refactor the code while keeping tests green
4. Ensure tests are properly documented
5. Verify that coverage remains above the threshold
