from glue.core.adhesive import AdhesiveType, ToolResult

def test_toolresult_backward_compat():
    # Old format
    tr = ToolResult(tool_name="foo", result=123)
    assert tr.tool_call_id == "call_foo"
    assert tr.content == 123
    assert tr.adhesive == AdhesiveType.GLUE
    # New format
    tr2 = ToolResult(tool_call_id="call_bar", content=456)
    assert tr2.tool_name == "bar"
    assert tr2.result == 456
    assert tr2.adhesive == AdhesiveType.GLUE
