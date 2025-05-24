from glue.core.app import GlueApp

def test_app_smoke():
    # Minimal config for smoke test
    config = {
        "name": "test-app",
        "models": [
            {
                "name": "m1",
                "provider": "openai",
                "model": "gpt-4",
            }
        ],
        "tools": [],
        "teams": [],
    }
    app = GlueApp(config)
    assert app.name == "test-app"
