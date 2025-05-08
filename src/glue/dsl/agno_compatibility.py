"""
Agno adapter compatibility module for the GLUE DSL.

This module provides functions to convert parsed DSL structures to Agno-compatible formats.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("glue.dsl.agno_compatibility")

def normalize_flow_operators(parsed_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes flow operators in a parsed GLUE configuration to be compatible with Agno.
    
    The GLUE DSL uses arrow operators (-> for PUSH, <- for PULL, >< for BIDIRECTIONAL,
    <> for REPEL) which need to be normalized to string constants for Agno.
    
    Args:
        parsed_config: Parsed configuration dictionary from the GLUE parser
        
    Returns:
        Normalized configuration dictionary
    """
    # Make a deep copy to avoid modifying the original
    result = parsed_config.copy()
    
    # Check if flows are defined
    if "flows" in result and isinstance(result["flows"], list):
        normalized_flows = []
        
        for flow in result["flows"]:
            if isinstance(flow, dict):
                normalized_flow = flow.copy()
                
                # Normalize flow types based on arrow operators
                if "type" in normalized_flow:
                    flow_type = normalized_flow["type"]
                    
                    # Map string operators to their normalized forms
                    if flow_type == "->":
                        normalized_flow["type"] = "PUSH"
                    elif flow_type == "<-":
                        normalized_flow["type"] = "PULL"
                    elif flow_type == "><":
                        normalized_flow["type"] = "BIDIRECTIONAL"
                    elif flow_type == "<>":
                        normalized_flow["type"] = "REPEL"
                    
                normalized_flows.append(normalized_flow)
                
        result["flows"] = normalized_flows
    
    return result

def process_flow_section(flows_config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process flow section from the DSL into Agno-compatible format.
    
    Args:
        flows_config: List of flow configurations from the parser
        
    Returns:
        Processed list of flow configurations
    """
    processed_flows = []
    
    for flow in flows_config:
        if isinstance(flow, dict):
            # Check for required fields
            if "source" in flow and "target" in flow:
                processed_flow = flow.copy()
                
                # Handle special case where 'pull' might be specified after <-
                if "type" in flow and flow["type"].lower() == "pull":
                    processed_flow["type"] = "PULL"
                
                processed_flows.append(processed_flow)
    
    return processed_flows
