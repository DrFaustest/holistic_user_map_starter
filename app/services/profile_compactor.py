from datetime import datetime, timedelta

from app.models.schemas import NodeType, UserMap, UserNode


class UserProfileCompactor:
    PROMOTION_EVIDENCE_THRESHOLD = 3
    PRUNE_AGE_DAYS = 21
    PRUNE_WEIGHT_THRESHOLD = 0.03
    PRUNE_CONFIDENCE_THRESHOLD = 0.03

    def compact(self, user_map: UserMap, now: datetime | None = None) -> dict[str, int]:
        now = now or datetime.utcnow()
        promoted = 0
        pruned = 0

        short_topic_ids = [
            node_id for node_id in user_map.nodes
            if node_id.startswith("topic_short_")
        ]

        for node_id in short_topic_ids:
            node = user_map.nodes.get(node_id)
            if node is None:
                continue
            if node.evidence_count < self.PROMOTION_EVIDENCE_THRESHOLD:
                continue
            promoted += self._promote_topic(user_map, node, now)

        stale_before = now - timedelta(days=self.PRUNE_AGE_DAYS)
        removable_ids = []
        for node_id, node in user_map.nodes.items():
            if node.last_updated > stale_before:
                continue
            if abs(node.weight) > self.PRUNE_WEIGHT_THRESHOLD:
                continue
            if node.confidence > self.PRUNE_CONFIDENCE_THRESHOLD:
                continue
            removable_ids.append(node_id)

        for node_id in removable_ids:
            del user_map.nodes[node_id]
            pruned += 1

        if promoted or pruned:
            user_map.updated_at = now

        return {
            "promoted_topics": promoted,
            "pruned_nodes": pruned,
        }

    def _promote_topic(self, user_map: UserMap, short_node: UserNode, now: datetime) -> int:
        topic_key = short_node.id.removeprefix("topic_short_")
        long_node_id = f"topic_long_{topic_key}"

        if long_node_id not in user_map.nodes:
            user_map.nodes[long_node_id] = UserNode(
                id=long_node_id,
                label=short_node.label.replace("Short-term", "Long-term"),
                type=NodeType.TOPIC_INTEREST,
                weight=0.0,
                confidence=0.0,
                metadata={"band": "long_term"},
            )

        long_node = user_map.nodes[long_node_id]
        long_node.weight = max(long_node.weight, min(1.0, short_node.weight * 0.85))
        long_node.confidence = max(long_node.confidence, min(1.0, short_node.confidence * 0.9))
        long_node.evidence_count = max(long_node.evidence_count, short_node.evidence_count)
        long_node.last_updated = now
        return 1