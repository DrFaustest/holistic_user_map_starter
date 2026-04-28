from datetime import datetime
from app.models.schemas import UserMap, UserNode, NodeType, InteractionAnalysis


class UserMapUpdater:
    LEARNING_RATE = 0.15
    BASE_DECAY_PER_DAY = 0.12
    SHORT_TOPIC_DECAY_PER_DAY = 0.12
    LONG_TOPIC_DECAY_PER_DAY = 0.04
    MIN_DECAY_MULTIPLIER = 0.2

    def apply_analysis(
        self,
        user_map: UserMap,
        analysis: InteractionAnalysis,
        now: datetime | None = None,
    ) -> list[UserNode]:
        now = now or datetime.utcnow()
        updated = []
        updated.extend(self.apply_decay(user_map, now=now))

        updated.append(
            self._update_node(
                user_map,
                node_id="pref_depth",
                label="Preference for deeper explanations",
                node_type=NodeType.COMMUNICATION_STYLE,
                delta=analysis.depth_preference_delta,
                now=now,
            )
        )

        updated.append(
            self._update_node(
                user_map,
                node_id="pref_concision",
                label="Preference for concise responses",
                node_type=NodeType.COMMUNICATION_STYLE,
                delta=analysis.concision_preference_delta,
                now=now,
            )
        )

        updated.append(
            self._update_node(
                user_map,
                node_id="signal_satisfaction",
                label="Recent satisfaction trend",
                node_type=NodeType.EMOTIONAL_SIGNAL,
                delta=analysis.satisfaction_score * self.LEARNING_RATE,
                now=now,
            )
        )

        updated.append(
            self._update_node(
                user_map,
                node_id="signal_confusion",
                label="Recent confusion trend",
                node_type=NodeType.EMOTIONAL_SIGNAL,
                delta=-analysis.confusion_score * self.LEARNING_RATE,
                now=now,
            )
        )

        updated.append(
            self._update_node(
                user_map,
                node_id="signal_engagement",
                label="Recent engagement trend",
                node_type=NodeType.TRUST_SIGNAL,
                delta=analysis.emotional_engagement_delta,
                now=now,
            )
        )

        for topic in analysis.detected_topics:
            updated.append(
                self._update_node(
                    user_map,
                    node_id=f"topic_short_{topic}",
                    label=f"Short-term interest in {topic.replace('_', ' ')}",
                    node_type=NodeType.TOPIC_INTEREST,
                    delta=0.08,
                    now=now,
                    metadata={"band": "short_term"},
                )
            )

        user_map.updated_at = now
        return updated

    def apply_decay(self, user_map: UserMap, now: datetime | None = None) -> list[UserNode]:
        now = now or datetime.utcnow()
        decayed_nodes = []

        for node in user_map.nodes.values():
            elapsed_seconds = (now - node.last_updated).total_seconds()
            if elapsed_seconds <= 0:
                continue

            elapsed_days = elapsed_seconds / 86400
            decay_rate = self._decay_rate(node)
            weight_multiplier = max(
                self.MIN_DECAY_MULTIPLIER,
                1.0 - decay_rate * elapsed_days,
            )
            confidence_multiplier = max(
                self.MIN_DECAY_MULTIPLIER,
                1.0 - (decay_rate * 0.6) * elapsed_days,
            )

            new_weight = node.weight * weight_multiplier
            new_confidence = node.confidence * confidence_multiplier

            if round(new_weight, 6) == round(node.weight, 6) and round(new_confidence, 6) == round(node.confidence, 6):
                continue

            node.weight = new_weight
            node.confidence = new_confidence
            node.last_updated = now
            decayed_nodes.append(node)

        if decayed_nodes:
            user_map.updated_at = now

        return decayed_nodes

    def _decay_rate(self, node: UserNode) -> float:
        if node.type == NodeType.TOPIC_INTEREST:
            retention_bonus = min(0.06, node.evidence_count * 0.008)
            if node.id.startswith("topic_long_"):
                return max(0.01, self.LONG_TOPIC_DECAY_PER_DAY - retention_bonus)
            return max(0.03, self.SHORT_TOPIC_DECAY_PER_DAY - retention_bonus)

        confidence_bonus = min(0.03, node.confidence * 0.03)
        return max(0.04, self.BASE_DECAY_PER_DAY - confidence_bonus)

    def _update_node(
        self,
        user_map: UserMap,
        node_id: str,
        label: str,
        node_type: NodeType,
        delta: float,
        now: datetime,
        metadata: dict[str, str] | None = None,
    ) -> UserNode:
        if node_id not in user_map.nodes:
            user_map.nodes[node_id] = UserNode(
                id=node_id,
                label=label,
                type=node_type,
                weight=0.0,
                confidence=0.0,
                metadata=metadata or {},
            )

        node = user_map.nodes[node_id]
        if metadata:
            node.metadata.update(metadata)
        node.weight = max(-1.0, min(1.0, node.weight + delta))
        node.confidence = min(1.0, node.confidence + 0.05)
        node.evidence_count += 1
        node.last_updated = now

        return node
