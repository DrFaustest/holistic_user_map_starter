from app.models.schemas import InteractionAnalysis, PromptAssembly, UserMap
from app.services.context_builder import PersonalizationContextBuilder


class PromptAssembler:
    LOW_CONFIDENCE_THRESHOLD = 0.18

    def __init__(self, context_builder: PersonalizationContextBuilder):
        self._context_builder = context_builder

    def assemble(
        self,
        user_map: UserMap,
        recent_interactions: list[dict[str, str | None]],
        max_evidence: int = 3,
    ) -> PromptAssembly:
        profile_version = user_map.updated_at.isoformat()
        context = self._context_builder.build_context(user_map)
        profile_confidence = self._profile_confidence(user_map)
        supporting_evidence: list[str] = []

        if profile_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            supporting_evidence = self._select_supporting_evidence(recent_interactions, max_evidence)

        instructions = self._build_prompt_instructions(context, supporting_evidence)

        return PromptAssembly(
            user_id=user_map.user_id,
            profile_version=profile_version,
            profile_confidence=profile_confidence,
            context=context,
            supporting_evidence=supporting_evidence,
            prompt_instructions=instructions,
        )

    def _profile_confidence(self, user_map: UserMap) -> float:
        active_nodes = [
            node for node in user_map.nodes.values()
            if abs(node.weight) >= 0.04 and node.confidence >= 0.03
        ]
        if not active_nodes:
            return 0.0
        ranked_scores = sorted(
            [abs(node.weight) * node.confidence for node in active_nodes],
            reverse=True,
        )[:4]
        return sum(ranked_scores) / len(ranked_scores)

    def _select_supporting_evidence(
        self,
        recent_interactions: list[dict[str, str | None]],
        max_evidence: int,
    ) -> list[str]:
        evidence_items = []
        for interaction in recent_interactions[:max_evidence]:
            analysis = InteractionAnalysis.model_validate_json(interaction["analysis_json"] or "{}")
            summary_parts = []
            if analysis.notes:
                summary_parts.append(analysis.notes[0])
            if analysis.detected_topics:
                summary_parts.append(f"topics: {', '.join(analysis.detected_topics[:2])}")
            if interaction.get("user_message"):
                summary_parts.append(f"user said: {interaction['user_message'][:80]}")
            if summary_parts:
                evidence_items.append("; ".join(summary_parts))
        return evidence_items

    def _build_prompt_instructions(self, context: str, supporting_evidence: list[str]) -> str:
        lines = [
            "Use the profile context as soft guidance.",
            context,
        ]
        if supporting_evidence:
            lines.append("Use the recent evidence only to disambiguate weak profile signals.")
            lines.extend(f"- {item}" for item in supporting_evidence)
        return "\n".join(lines)