import pytest
from glue.core.teams import Team

def test_team_empty_name():
    t = Team(name="", description="desc", members=[])
    assert t.name == ""
    assert t.description == "desc"
    assert t.members == []

def test_team_with_members():
    t = Team(name="t2", description="desc2", members=["alice", "bob"])
    assert t.name == "t2"
    assert t.members == ["alice", "bob"]

@pytest.mark.parametrize("bad_members", [123, "notalist", {"a":1}])
def test_team_invalid_members(bad_members):
    with pytest.raises(Exception):
        Team(name="bad", members=bad_members)

def test_team_none_members():
    # None is allowed and should not raise
    Team(name="none_ok", members=None)
