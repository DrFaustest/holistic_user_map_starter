import re
from app.models.schemas import InteractionAnalysis


class InteractionAnalyzer:
    """
    Rule-based placeholder.

    Later, replace this with:
    - transformer sentiment model
    - emotion classifier
    - topic classifier
    - LLM-based structured extraction
    - reinforcement signal model
    """

    POSITIVE_PATTERNS = [
        r"\bthank you\b",
        r"\bthanks\b",
        r"\bhelpful\b",
        r"\bthat makes sense\b",
        r"\bexactly\b",
        r"\byes\b",
    ]

    NEGATIVE_PATTERNS = [
        r"\bnot what i meant\b",
        r"\bthat's not what i meant\b",
        r"\bmissed the point\b",
        r"\bwrong\b",
        r"\bconfusing\b",
        r"\bi don't understand\b",
        r"\btoo vague\b",
        r"\btoo shallow\b",
    ]

    DEPTH_PATTERNS = [
        r"\bdeeper\b",
        r"\bmore detail\b",
        r"\bexplain why\b",
        r"\bthorough\b",
        r"\btechnical\b",
        r"\bmathematical\b",
    ]

    CONCISION_PATTERNS = [
        r"\bshort\b",
        r"\bquick\b",
        r"\bconcise\b",
        r"\bbrief\b",
        r"\btoo long\b",
    ]

    DETAILED_RESPONSE_PATTERNS = [
        r"\bdetailed\b",
        r"\bthorough\b",
        r"\bstep-by-step\b",
        r"\bin depth\b",
        r"\bin-depth\b",
    ]

    BRIEF_RESPONSE_PATTERNS = [
        r"\bshort\b",
        r"\bbrief\b",
        r"\bquick\b",
        r"\bconcise\b",
        r"\bone-line\b",
    ]

    TOPIC_KEYWORDS = {
        "ai_personalization": ["personalized ai", "large language model", "llm", "user map", "memory"],
        "education": ["knowledge tracing", "student", "learning", "education", "teacher"],
        "graph_modeling": ["map", "node", "edge", "graph", "heterogeneous", "temporal"],
        "software_architecture": ["layer", "filter", "context", "pipeline", "integration"],
    }

    def analyze(
        self,
        user_message: str,
        assistant_response: str,
        explicit_feedback: str | None = None,
    ) -> InteractionAnalysis:
        text = " ".join(
            part for part in [user_message, explicit_feedback or ""] if part
        ).lower()
        assistant_text = assistant_response.lower()

        positive_hits = self._count_hits(text, self.POSITIVE_PATTERNS)
        negative_hits = self._count_hits(text, self.NEGATIVE_PATTERNS)
        depth_hits = self._count_hits(text, self.DEPTH_PATTERNS)
        concision_hits = self._count_hits(text, self.CONCISION_PATTERNS)
        detailed_response_hits = self._count_hits(assistant_text, self.DETAILED_RESPONSE_PATTERNS)
        brief_response_hits = self._count_hits(assistant_text, self.BRIEF_RESPONSE_PATTERNS)

        satisfaction = max(-1.0, min(1.0, (positive_hits - negative_hits) / 2.0))
        confusion = min(1.0, negative_hits / 2.0)

        depth_delta = min(0.25, depth_hits * 0.1)
        concision_delta = min(0.25, concision_hits * 0.1)

        if "too long" in text:
            concision_delta += 0.15
            depth_delta -= 0.05

        if "too vague" in text or "too shallow" in text:
            depth_delta += 0.15
            concision_delta -= 0.05

        if negative_hits and brief_response_hits:
            depth_delta += 0.10
            concision_delta -= 0.05

        if positive_hits and detailed_response_hits:
            depth_delta += 0.05

        if positive_hits and brief_response_hits:
            concision_delta += 0.05

        topics = self._detect_topics(" ".join(part for part in [text, assistant_text] if part))

        notes = []
        if satisfaction > 0:
            notes.append("User response suggests the answer was useful.")
        if satisfaction < 0:
            notes.append("User response suggests the answer missed intent.")
        if negative_hits and brief_response_hits:
            notes.append("The prior answer was likely too brief for the user's needs.")
        if positive_hits and detailed_response_hits:
            notes.append("The user responded well to a detailed answer.")
        if depth_delta > 0:
            notes.append("User appears to prefer deeper explanations.")
        if concision_delta > 0:
            notes.append("User appears to prefer concise answers.")

        return InteractionAnalysis(
            satisfaction_score=satisfaction,
            confusion_score=confusion,
            depth_preference_delta=max(-0.25, min(0.25, depth_delta)),
            concision_preference_delta=max(-0.25, min(0.25, concision_delta)),
            emotional_engagement_delta=max(0.0, min(0.2, positive_hits * 0.05 + detailed_response_hits * 0.02)),
            detected_topics=topics,
            notes=notes,
        )

    def _count_hits(self, text: str, patterns: list[str]) -> int:
        return sum(1 for pattern in patterns if re.search(pattern, text))

    def _detect_topics(self, text: str) -> list[str]:
        topics = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        return topics
