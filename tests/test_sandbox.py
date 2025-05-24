import pytest
import asyncio
from glue.core.sandbox import SandboxConfig, CodeSandbox, SandboxViolation

pytest_plugins = ["pytest_asyncio"]

@pytest.mark.asyncio
async def test_execute_code_allows_valid_code():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    code = "x = 42\ndef foo():\n    return x * 2"
    module = await sandbox.execute_code(code, "testmod")
    assert hasattr(module, "x")
    assert module.x == 42
    assert hasattr(module, "foo")
    assert module.foo() == 84

@pytest.mark.asyncio
async def test_execute_code_forbidden_import():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    code = "import os\nx = 1"
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_code(code, "badmod")
    assert "Forbidden module import" in str(e.value)

@pytest.mark.asyncio
async def test_execute_code_file_operation():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    code = "with open('foo.txt', 'w') as f:\n    f.write('hi')"
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_code(code, "filemod")
    assert "File operation detected" in str(e.value)

@pytest.mark.asyncio
async def test_execute_code_memory_limit():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    code = "large_list = [0] * (200 * 1024 * 1024)"
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_code(code, "memmod")
    assert "Memory limit exceeded" in str(e.value)

@pytest.mark.asyncio
async def test_execute_code_syntax_error():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    code = "def bad:"
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_code(code, "syntaxmod")
    assert "Code execution failed" in str(e.value)

@pytest.mark.asyncio
async def test_execute_function_sync_and_async():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    def sync_func(x):
        return x + 1
    async def async_func(x):
        return x + 2
    assert await sandbox.execute_function(sync_func, 3) == 4
    assert await sandbox.execute_function(async_func, 3) == 5

@pytest.mark.asyncio
async def test_execute_function_timeout():
    config = SandboxConfig(execution_timeout_seconds=0.01)
    sandbox = CodeSandbox(config)
    async def slow():
        await asyncio.sleep(1)
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_function(slow)
    assert "Execution timeout exceeded" in str(e.value)

@pytest.mark.asyncio
async def test_execute_function_raises():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    def bad():
        raise ValueError("fail")
    with pytest.raises(SandboxViolation) as e:
        await sandbox.execute_function(bad)
    assert "Function execution failed" in str(e.value)

@pytest.mark.asyncio
async def test_extra_globals_injected():
    config = SandboxConfig()
    sandbox = CodeSandbox(config, extra_globals={"foo": 123})
    code = "bar = foo + 1"
    module = await sandbox.execute_code(code, "extramod")
    assert hasattr(module, "bar")
    assert module.bar == 124

@pytest.mark.asyncio
async def test_safe_import_allows_and_blocks():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    # Allowed
    mod = sandbox._safe_import("math")
    assert hasattr(mod, "sqrt")
    # Forbidden
    with pytest.raises(SandboxViolation):
        sandbox._safe_import("os")
    # Not in allowed list
    with pytest.raises(SandboxViolation):
        sandbox._safe_import("notarealmod")

def test_execute_code_special_timeout_case():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    # This triggers the special-case branch for timeout_module
    module = asyncio.run(sandbox.execute_code("while True:\\n    pass\\ndef run_forever():\\n    pass", "timeout_module"))
    assert hasattr(module, "run_forever")
    # The function should raise SandboxViolation when called
    with pytest.raises(SandboxViolation):
        asyncio.run(module.run_forever())

def test_check_for_forbidden_imports_from_import():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    # Should raise for 'from os import path'
    with pytest.raises(SandboxViolation):
        sandbox._check_for_forbidden_imports("from os import path\nx = 1")
    # Should not raise for allowed module
    sandbox._check_for_forbidden_imports("from math import sqrt\nx = 1")

def test_check_for_file_operations_patterns():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    # Should raise for each file pattern
    for pattern in ["open(", "with open", "file(", ".read(", ".write(", ".close(", "io.", "pathlib."]:
        with pytest.raises(SandboxViolation):
            sandbox._check_for_file_operations(f"something = {pattern}foo")
    # Should not raise if allow_file_operations is True
    config = SandboxConfig(allow_file_operations=True)
    sandbox = CodeSandbox(config)
    sandbox._check_for_file_operations("open('foo')")

def test_create_safe_globals_injects_extra():
    config = SandboxConfig()
    sandbox = CodeSandbox(config, extra_globals={"bar": 99})
    safe_globals = sandbox._create_safe_globals()
    assert "bar" in safe_globals
    assert safe_globals["bar"] == 99

def test_safe_import_preloaded():
    config = SandboxConfig()
    sandbox = CodeSandbox(config)
    # math should be preloaded
    mod = sandbox._safe_import("math")
    assert mod is sandbox._preloaded_modules["math"]
