"""AnonTokyo simulation policy and value heads."""

from __future__ import annotations

from anon_tokyo.simulation.agent_centric.policy_head import AgentCentricPolicyHead


class AnonTokyoPolicyHead(AgentCentricPolicyHead):
    """Bounded Beta policy head for AnonTokyo simulation policies."""


__all__ = ["AnonTokyoPolicyHead"]
