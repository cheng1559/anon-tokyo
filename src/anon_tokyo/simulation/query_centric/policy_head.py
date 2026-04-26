"""Query-centric simulation policy and value heads."""

from __future__ import annotations

from anon_tokyo.simulation.agent_centric.policy_head import AgentCentricPolicyHead


class QueryCentricPolicyHead(AgentCentricPolicyHead):
    """Bounded Beta policy head for query-centric simulation policies."""


__all__ = ["QueryCentricPolicyHead"]
