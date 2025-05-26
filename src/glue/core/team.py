"""
Team module for the GLUE framework.

This module is a compatibility layer that re-exports the Team class from
the teams module to maintain backward compatibility with existing tests.
"""

from .teams import GlueTeam

# Re-export GlueTeam as Team class for backward compatibility
Team = GlueTeam
__all__ = ["Team"]
