from app.models.schemas import UserMap


class PersonalizationContextBuilder:
    """
    Converts the graph/map into compact LLM context.

    This is the token-saving layer.
    Instead of injecting raw history, inject only the active high-confidence traits.
    """

    def build_context(self, user_map: UserMap, max_nodes: int = 6) -> str:
        ranked_nodes = sorted(
            user_map.nodes.values(),
            key=lambda n: (abs(n.weight) * n.confidence, n.evidence_count),
            reverse=True,
        )

        active_nodes = [
            node for node in ranked_nodes
            if abs(node.weight) >= 0.04 and node.confidence >= 0.03
        ][:max_nodes]

        if not active_nodes:
            return "No stable personalization context is available yet."

        communication_items = self._format_nodes(active_nodes, prefix="pref_")
        topic_items = self._format_topics(active_nodes)
        signal_items = self._format_signal_nodes(active_nodes)

        lines = ["Personalization context:"]

        if communication_items:
            lines.append(f"Response style: {', '.join(communication_items)}.")

        if topic_items:
            lines.append(f"Likely recurring topics: {', '.join(topic_items)}.")

        if signal_items:
            lines.append(f"Interaction signals: {', '.join(signal_items)}.")

        return "\n".join(lines)

    def _format_nodes(self, nodes, prefix: str) -> list[str]:
        items = []
        for node in nodes:
            if not node.id.startswith(prefix):
                continue
            items.append(f"{node.label.lower()} {self._describe_signal_level(node.weight)}")
        return items

    def _format_topics(self, nodes) -> list[str]:
        items = []
        for node in nodes:
            if not node.id.startswith("topic_"):
                continue
            topic_label = node.label.lower()
            if node.id.startswith("topic_short_"):
                items.append(f"{topic_label} {self._describe_signal_level(node.weight)}")
            elif node.id.startswith("topic_long_"):
                items.append(f"{topic_label} {self._describe_signal_level(node.weight)}")
        return items

    def _format_signal_nodes(self, nodes) -> list[str]:
        items = []
        for node in nodes:
            if not node.id.startswith("signal_"):
                continue
            items.append(f"{node.label.lower()} {self._describe_signal_level(node.weight)}")
        return items

    def _describe_signal_level(self, weight: float) -> str:
        if weight >= 0.18:
            return "strong"
        if weight >= 0.05:
            return "moderate"
        if weight <= -0.18:
            return "suppressed"
        if weight <= -0.05:
            return "reduced"
        return "mixed"
