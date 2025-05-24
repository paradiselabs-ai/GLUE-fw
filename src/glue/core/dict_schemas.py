# Pure dictionary approach - maximum smolagents compatibility
# This is the most lightweight approach that smolagents prefers

def create_model_config_dict(name, provider, processed_config):
    """Create model config as a simple dictionary with defaults"""
    return {
        "name": name,
        "provider": provider,
        "model": processed_config.get("model", provider),
        "temperature": processed_config.get("temperature", 0.7),
        "max_tokens": processed_config.get("max_tokens", 2048),
        "description": processed_config.get("description", ""),
        "api_key": processed_config.get("api_key"),
        "api_params": processed_config.get("api_params", {}),
        "provider_class": processed_config.get("provider_class")
    }

def validate_model_config(config_dict):
    """Basic validation for model config dictionary"""
    required_fields = ["name", "provider", "model"]
    for field in required_fields:
        if not config_dict.get(field):
            raise ValueError(f"Required field '{field}' is missing or empty")
    
    temp = config_dict.get("temperature", 0.7)
    if not (0.0 <= temp <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")
    
    max_tokens = config_dict.get("max_tokens", 2048)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    
    return config_dict
