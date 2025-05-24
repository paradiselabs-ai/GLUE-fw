from glue.core.teams import Team

def test_team_smoke():
    t = Team(name="t1")
    assert t.name == "t1"
    # Add a dummy member
    class Dummy:
        def __init__(self, name):
            self.name = name
    m = Dummy("bob")
    t.add_member_sync(m)
    assert "bob" in t.models
