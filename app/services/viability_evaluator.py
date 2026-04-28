import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app import __version__
from app.models.schemas import (
    CostBenefitEvaluation,
    EvaluationArtifact,
    EvaluationReportMetadata,
    ViabilityExperimentResult,
    ViabilityScenarioResult,
    UserMap,
)
from app.services.analyzer import InteractionAnalyzer
from app.services.map_updater import UserMapUpdater
from app.services.profile_compactor import UserProfileCompactor
from app.services.settings import LLMSettings, load_llm_settings
from app.services.token_cost import ProviderTokenCostEstimator


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    source: str
    preferred_style: str
    preferred_topics: tuple[str, ...]
    request_template: str
    positive_feedback: str
    negative_feedback_style: str
    negative_feedback_topic_template: str


class PersonalizationViabilityEvaluator:
    WORTH_IT_THRESHOLD = 0.01

    def __init__(
        self,
        analyzer: InteractionAnalyzer,
        updater: UserMapUpdater,
        profile_compactor: UserProfileCompactor,
        settings: LLMSettings | None = None,
        report_dir: Path | None = None,
    ):
        self._analyzer = analyzer
        self._updater = updater
        self._profile_compactor = profile_compactor
        self._settings = settings or load_llm_settings()
        self._cost_estimator = ProviderTokenCostEstimator(self._settings)
        self._report_dir = report_dir or Path(__file__).resolve().parents[2] / "data" / "reports"
        self._scenarios = self._default_scenarios() + self._human_labeled_scenarios()

    def run_experiment(self, rounds_per_user: int = 4) -> ViabilityExperimentResult:
        scenario_results: list[ViabilityScenarioResult] = []

        for scenario in self._scenarios:
            baseline_alignment_scores, baseline_satisfaction_scores, baseline_token_costs = self._run_condition(
                scenario,
                rounds_per_user,
                personalized=False,
            )
            personalized_alignment_scores, personalized_satisfaction_scores, personalized_token_costs = self._run_condition(
                scenario,
                rounds_per_user,
                personalized=True,
            )

            baseline_alignment = self._mean(baseline_alignment_scores)
            personalized_alignment = self._mean(personalized_alignment_scores)
            baseline_satisfaction = self._mean(baseline_satisfaction_scores)
            personalized_satisfaction = self._mean(personalized_satisfaction_scores)
            baseline_token_cost = self._mean(baseline_token_costs, digits=8)
            personalized_token_cost = self._mean(personalized_token_costs, digits=8)
            token_cost_delta = round(personalized_token_cost - baseline_token_cost, 8)
            alignment_lift = round(personalized_alignment - baseline_alignment, 4)
            satisfaction_lift = round(personalized_satisfaction - baseline_satisfaction, 4)
            alignment_lift_per_token_delta = round(
                alignment_lift / token_cost_delta,
                4,
            ) if token_cost_delta > 0 else alignment_lift

            scenario_results.append(
                ViabilityScenarioResult(
                    scenario_name=scenario.name,
                    scenario_source=scenario.source,
                    baseline_mean_alignment=baseline_alignment,
                    personalized_mean_alignment=personalized_alignment,
                    baseline_mean_satisfaction=baseline_satisfaction,
                    personalized_mean_satisfaction=personalized_satisfaction,
                    baseline_mean_token_cost=baseline_token_cost,
                    personalized_mean_token_cost=personalized_token_cost,
                    alignment_lift=alignment_lift,
                    satisfaction_lift=satisfaction_lift,
                    token_cost_delta=token_cost_delta,
                    alignment_lift_per_token_delta=alignment_lift_per_token_delta,
                )
            )

        result = self._aggregate_results(scenario_results, rounds_per_user)
        result.exported_artifacts = self.export_report(result, formats=("json", "csv"))
        return result

    def export_report(
        self,
        result: ViabilityExperimentResult,
        formats: tuple[str, ...] = ("json",),
    ) -> list[EvaluationArtifact]:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = result.report_metadata.generated_at.strftime("%Y%m%dT%H%M%SZ")
        revision_suffix = result.report_metadata.git_revision or result.report_metadata.app_version.replace(".", "_")
        artifacts: list[EvaluationArtifact] = []

        if "json" in formats:
            json_path = self._report_dir / f"viability_report_{revision_suffix}_{timestamp}.json"
            json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
            artifacts.append(EvaluationArtifact(format="json", path=str(json_path)))

        if "csv" in formats:
            csv_path = self._report_dir / f"viability_report_{revision_suffix}_{timestamp}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "generated_at",
                        "git_revision",
                        "app_version",
                        "llm_provider",
                        "llm_model",
                        "tokenizer_name",
                        "scenario_name",
                        "scenario_source",
                        "baseline_mean_alignment",
                        "personalized_mean_alignment",
                        "baseline_mean_satisfaction",
                        "personalized_mean_satisfaction",
                        "baseline_mean_token_cost",
                        "personalized_mean_token_cost",
                        "alignment_lift",
                        "satisfaction_lift",
                        "token_cost_delta",
                        "alignment_lift_per_token_delta",
                    ],
                )
                writer.writeheader()
                for scenario_result in result.scenario_results:
                    writer.writerow(
                        {
                            "generated_at": result.report_metadata.generated_at.isoformat(),
                            "git_revision": result.report_metadata.git_revision,
                            "app_version": result.report_metadata.app_version,
                            "llm_provider": result.report_metadata.llm_provider,
                            "llm_model": result.report_metadata.llm_model,
                            "tokenizer_name": result.report_metadata.tokenizer_name,
                            **scenario_result.model_dump(),
                        }
                    )
            artifacts.append(EvaluationArtifact(format="csv", path=str(csv_path)))

        return artifacts

    def _aggregate_results(
        self,
        scenario_results: list[ViabilityScenarioResult],
        rounds_per_user: int,
    ) -> ViabilityExperimentResult:
        baseline_mean_alignment = self._mean([result.baseline_mean_alignment for result in scenario_results])
        personalized_mean_alignment = self._mean([result.personalized_mean_alignment for result in scenario_results])
        baseline_mean_satisfaction = self._mean([result.baseline_mean_satisfaction for result in scenario_results])
        personalized_mean_satisfaction = self._mean([result.personalized_mean_satisfaction for result in scenario_results])
        baseline_mean_token_cost = self._mean([result.baseline_mean_token_cost for result in scenario_results], digits=8)
        personalized_mean_token_cost = self._mean([result.personalized_mean_token_cost for result in scenario_results], digits=8)
        mean_alignment_lift = round(personalized_mean_alignment - baseline_mean_alignment, 4)
        mean_satisfaction_lift = round(personalized_mean_satisfaction - baseline_mean_satisfaction, 4)
        mean_token_cost_delta = round(personalized_mean_token_cost - baseline_mean_token_cost, 8)
        alignment_lift_per_token_delta = round(
            mean_alignment_lift / mean_token_cost_delta,
            4,
        ) if mean_token_cost_delta > 0 else mean_alignment_lift
        satisfaction_lift_per_token_delta = round(
            mean_satisfaction_lift / mean_token_cost_delta,
            4,
        ) if mean_token_cost_delta > 0 else mean_satisfaction_lift

        return ViabilityExperimentResult(
            hypothesis="A persistent user-profile model improves response alignment and simulated user satisfaction compared with a no-memory baseline.",
            primary_metric="mean preference-alignment score",
            secondary_metric="mean simulated satisfaction score",
            report_metadata=self._build_report_metadata(),
            rounds_per_user=rounds_per_user,
            scenario_count=len(self._scenarios),
            baseline_mean_alignment=baseline_mean_alignment,
            personalized_mean_alignment=personalized_mean_alignment,
            baseline_mean_satisfaction=baseline_mean_satisfaction,
            personalized_mean_satisfaction=personalized_mean_satisfaction,
            baseline_mean_token_cost=baseline_mean_token_cost,
            personalized_mean_token_cost=personalized_mean_token_cost,
            mean_alignment_lift=mean_alignment_lift,
            mean_satisfaction_lift=mean_satisfaction_lift,
            cost_benefit=CostBenefitEvaluation(
                baseline_mean_token_cost=baseline_mean_token_cost,
                personalized_mean_token_cost=personalized_mean_token_cost,
                mean_token_cost_delta=mean_token_cost_delta,
                alignment_lift_per_token_delta=alignment_lift_per_token_delta,
                satisfaction_lift_per_token_delta=satisfaction_lift_per_token_delta,
                worth_it_threshold=self.WORTH_IT_THRESHOLD,
                worth_it=alignment_lift_per_token_delta > self.WORTH_IT_THRESHOLD,
            ),
            supports_viability=mean_alignment_lift > 0.15 and mean_satisfaction_lift > 0.15,
            methodology=[
                "Use a fixed mixed panel of synthetic and human-labeled users with hidden style and topic preferences.",
                "Run repeated interactions under two conditions: generic baseline and persistence-driven personalization.",
                "Update the personalized condition with the same analyzer and map updater used by the application.",
                "Score both conditions with the same deterministic judge for alignment, satisfaction, and approximate token cost.",
            ],
            scenario_results=scenario_results,
            exported_artifacts=[],
        )

    def _run_condition(
        self,
        scenario: EvaluationScenario,
        rounds_per_user: int,
        personalized: bool,
    ) -> tuple[list[float], list[float], list[float]]:
        user_map = UserMap(user_id=f"eval-{scenario.name}-{'personalized' if personalized else 'baseline'}")
        alignment_scores: list[float] = []
        satisfaction_scores: list[float] = []
        token_costs: list[float] = []

        for round_index in range(rounds_per_user):
            user_message = scenario.request_template.format(topic=scenario.preferred_topics[round_index % len(scenario.preferred_topics)])
            if personalized:
                prompt_instructions = self._personalized_prompt_instructions(user_map, scenario)
            else:
                prompt_instructions = self._baseline_prompt_instructions(scenario)
            if personalized:
                assistant_response = self._personalized_response(user_map, scenario, round_index)
            else:
                assistant_response = self._baseline_response(scenario, round_index)

            alignment_score = self._alignment_score(assistant_response, scenario)
            satisfaction_score = self._simulated_satisfaction(alignment_score)
            token_cost = self._token_cost(prompt_instructions, user_message, assistant_response)
            alignment_scores.append(alignment_score)
            satisfaction_scores.append(satisfaction_score)
            token_costs.append(token_cost)

            explicit_feedback = self._simulated_feedback(assistant_response, scenario)
            analysis = self._analyzer.analyze(
                user_message=user_message,
                assistant_response=assistant_response,
                explicit_feedback=explicit_feedback,
            )

            if personalized:
                self._updater.apply_analysis(user_map, analysis)
                self._profile_compactor.compact(user_map)

        return alignment_scores, satisfaction_scores, token_costs

    def _baseline_response(self, scenario: EvaluationScenario, round_index: int) -> str:
        topic = scenario.preferred_topics[round_index % len(scenario.preferred_topics)]
        return (
            f"Here is a general overview of {topic}. "
            "It covers the basics with a balanced explanation and a simple takeaway."
        )

    def _baseline_prompt_instructions(self, scenario: EvaluationScenario) -> str:
        return (
            "You are a helpful assistant. "
            f"Answer with a balanced explanation about {scenario.preferred_topics[0]} and keep the tone neutral."
        )

    def _personalized_prompt_instructions(self, user_map: UserMap, scenario: EvaluationScenario) -> str:
        preferred_topic = self._select_topic(user_map, scenario, 0)
        if "pref_depth" in user_map.nodes and user_map.nodes["pref_depth"].weight > 0:
            style_instruction = "Favor a detailed technical explanation."
        elif "pref_concision" in user_map.nodes and user_map.nodes["pref_concision"].weight > 0:
            style_instruction = "Favor a brief concise explanation."
        else:
            style_instruction = "Start balanced while adapting to emerging preferences."

        tracked_topics = [
            node.label.replace("Short-term interest in ", "").replace("Long-term interest in ", "")
            for node in user_map.nodes.values()
            if node.id.startswith("topic_")
        ]
        topic_summary = ", ".join(tracked_topics[:3]) if tracked_topics else preferred_topic
        return (
            "You are a personalized assistant using persisted profile memory. "
            "Consult the inferred user profile before responding, adapt the response style to the stored preferences, "
            "and preserve continuity with recurring interests when they are relevant. "
            f"{style_instruction} "
            f"Current likely interests: {topic_summary}. "
            "Treat this profile summary as additional prompt context that is unavailable to the baseline condition."
        )

    def _personalized_response(self, user_map: UserMap, scenario: EvaluationScenario, round_index: int) -> str:
        topic = self._select_topic(user_map, scenario, round_index)
        depth_weight = user_map.nodes.get("pref_depth").weight if "pref_depth" in user_map.nodes else 0.0
        concision_weight = user_map.nodes.get("pref_concision").weight if "pref_concision" in user_map.nodes else 0.0

        if depth_weight > concision_weight + 0.05:
            return f"Here is a detailed technical explanation of {topic}, including step-by-step reasoning, tradeoffs, and implementation details."
        if concision_weight > depth_weight + 0.05:
            return f"Here is a brief concise summary of {topic} with the key takeaway only."

        return f"Here is an overview of {topic} with the main concepts and practical tradeoffs."

    def _select_topic(self, user_map: UserMap, scenario: EvaluationScenario, round_index: int) -> str:
        long_term_topics = [node for node in user_map.nodes.values() if node.id.startswith("topic_long_")]
        short_term_topics = [node for node in user_map.nodes.values() if node.id.startswith("topic_short_")]
        ranked_topics = sorted(
            long_term_topics + short_term_topics,
            key=lambda node: (abs(node.weight) * node.confidence, node.evidence_count),
            reverse=True,
        )
        if ranked_topics:
            return ranked_topics[0].label.replace("Short-term interest in ", "").replace("Long-term interest in ", "")
        return scenario.preferred_topics[round_index % len(scenario.preferred_topics)]

    def _alignment_score(self, assistant_response: str, scenario: EvaluationScenario) -> float:
        response = assistant_response.lower()
        style_score = 0.0

        if scenario.preferred_style == "deep":
            if any(keyword in response for keyword in ["detailed", "technical", "step-by-step", "tradeoffs"]):
                style_score = 1.0
        elif scenario.preferred_style == "brief":
            if any(keyword in response for keyword in ["brief", "concise", "key takeaway"]):
                style_score = 1.0

        topic_hits = sum(1 for topic in scenario.preferred_topics if topic in response)
        topic_score = topic_hits / len(scenario.preferred_topics)
        return round((style_score * 0.7) + (topic_score * 0.3), 4)

    def _simulated_satisfaction(self, alignment_score: float) -> float:
        return round((alignment_score * 2.0) - 1.0, 4)

    def _simulated_feedback(self, assistant_response: str, scenario: EvaluationScenario) -> str:
        response = assistant_response.lower()
        style_match = False

        if scenario.preferred_style == "deep":
            style_match = any(keyword in response for keyword in ["detailed", "technical", "step-by-step", "tradeoffs"])
            if not style_match:
                return scenario.negative_feedback_style
        if scenario.preferred_style == "brief":
            style_match = any(keyword in response for keyword in ["brief", "concise", "key takeaway"])
            if not style_match:
                return scenario.negative_feedback_style

        missing_topics = [topic for topic in scenario.preferred_topics if topic not in response]
        if missing_topics:
            return scenario.negative_feedback_topic_template.format(topic=missing_topics[0])

        return scenario.positive_feedback

    def _token_cost(self, prompt_instructions: str, user_message: str, assistant_response: str) -> float:
        cost_estimate = self._cost_estimator.estimate_chat_cost(
            system_prompt=prompt_instructions,
            user_message=user_message,
            assistant_response=assistant_response,
        )
        return cost_estimate.estimated_cost_usd

    def _build_report_metadata(self) -> EvaluationReportMetadata:
        return EvaluationReportMetadata(
            generated_at=datetime.utcnow(),
            git_revision=self._resolve_git_revision(),
            app_version=__version__,
            llm_provider=self._settings.provider,
            llm_model=self._settings.model,
            tokenizer_name=self._cost_estimator.tokenizer_name,
        )

    def _resolve_git_revision(self) -> str | None:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None
        revision = result.stdout.strip()
        return revision or None

    def _default_scenarios(self) -> list[EvaluationScenario]:
        return [
            EvaluationScenario(
                name="deep_llm_builder",
                source="synthetic",
                preferred_style="deep",
                preferred_topics=("llm", "memory"),
                request_template="Help me understand {topic} tradeoffs for an AI assistant.",
                positive_feedback="Thanks, that matched the level of detail and topic focus I wanted.",
                negative_feedback_style="That was too shallow. I wanted a deeper, more technical explanation.",
                negative_feedback_topic_template="Please focus more on {topic}.",
            ),
            EvaluationScenario(
                name="brief_architect",
                source="synthetic",
                preferred_style="brief",
                preferred_topics=("context", "pipeline"),
                request_template="What matters most about {topic} in a production assistant?",
                positive_feedback="Thanks, that matched the level of detail and topic focus I wanted.",
                negative_feedback_style="That was too long. Please keep it brief and concise.",
                negative_feedback_topic_template="Please focus more on {topic}.",
            ),
            EvaluationScenario(
                name="deep_education_user",
                source="synthetic",
                preferred_style="deep",
                preferred_topics=("learning", "education"),
                request_template="Explain how {topic} relates to user modeling.",
                positive_feedback="Thanks, that matched the level of detail and topic focus I wanted.",
                negative_feedback_style="That was too shallow. I wanted a deeper, more technical explanation.",
                negative_feedback_topic_template="Please focus more on {topic}.",
            ),
            EvaluationScenario(
                name="brief_graph_user",
                source="synthetic",
                preferred_style="brief",
                preferred_topics=("graph", "node"),
                request_template="Give me the key idea behind {topic} for personalization.",
                positive_feedback="Thanks, that matched the level of detail and topic focus I wanted.",
                negative_feedback_style="That was too long. Please keep it brief and concise.",
                negative_feedback_topic_template="Please focus more on {topic}.",
            ),
        ]

    def _human_labeled_scenarios(self) -> list[EvaluationScenario]:
        dataset_path = Path(__file__).resolve().parents[1] / "evaluation" / "human_scenarios.json"
        payload = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [
            EvaluationScenario(
                name=item["name"],
                source=item["source"],
                preferred_style=item["preferred_style"],
                preferred_topics=tuple(item["preferred_topics"]),
                request_template=item["request_template"],
                positive_feedback=item["positive_feedback"],
                negative_feedback_style=item["negative_feedback_style"],
                negative_feedback_topic_template=item["negative_feedback_topic_template"],
            )
            for item in payload
        ]

    def _mean(self, values: list[float], digits: int = 4) -> float:
        if not values:
            return 0.0
        return round(sum(values) / len(values), digits)