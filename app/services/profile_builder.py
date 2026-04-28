from app.models.schemas import NodeType, ProfileTrait, UserMap, UserProfile


class UserProfileBuilder:
    """
    Builds a structured profile view from the evolving user map.

    This keeps the API focused on what the assistant should understand about the
    user rather than exposing only raw node data.
    """

    def build_profile(self, user_map: UserMap, max_traits_per_section: int = 4) -> UserProfile:
        return UserProfile(
            user_id=user_map.user_id,
            communication_preferences=self._traits_for_type(
                user_map,
                NodeType.COMMUNICATION_STYLE,
                max_traits_per_section,
            ),
            cognitive_preferences=self._traits_for_type(
                user_map,
                NodeType.COGNITIVE_STYLE,
                max_traits_per_section,
            ),
            short_term_topic_interests=self._traits_for_type(
                user_map,
                NodeType.TOPIC_INTEREST,
                max_traits_per_section,
                id_prefix="topic_short_",
            ),
            long_term_topic_interests=self._traits_for_type(
                user_map,
                NodeType.TOPIC_INTEREST,
                max_traits_per_section,
                id_prefix="topic_long_",
            ),
            emotional_signals=self._traits_for_type(
                user_map,
                NodeType.EMOTIONAL_SIGNAL,
                max_traits_per_section,
            ),
            trust_signals=self._traits_for_type(
                user_map,
                NodeType.TRUST_SIGNAL,
                max_traits_per_section,
            ),
            updated_at=user_map.updated_at,
        )

    def _traits_for_type(
        self,
        user_map: UserMap,
        node_type: NodeType,
        max_traits: int,
        id_prefix: str | None = None,
    ) -> list[ProfileTrait]:
        matching_nodes = [
            node for node in user_map.nodes.values()
            if node.type == node_type and (id_prefix is None or node.id.startswith(id_prefix))
        ]

        ranked_nodes = sorted(
            matching_nodes,
            key=lambda node: (abs(node.weight) * node.confidence, node.evidence_count),
            reverse=True,
        )[:max_traits]

        return [
            ProfileTrait(
                id=node.id,
                label=node.label,
                weight=node.weight,
                confidence=node.confidence,
                evidence_count=node.evidence_count,
            )
            for node in ranked_nodes
        ]