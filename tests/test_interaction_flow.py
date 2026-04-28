import csv
from io import StringIO
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from app import __version__
from app.cli import main as cli_main
from app.models.schemas import NodeType, UserMap, UserNode
from app.services.analyzer import InteractionAnalyzer
from app.services.chat_service import PersonalizedChatService
from app.services.map_updater import UserMapUpdater
from app.services.context_builder import PersonalizationContextBuilder
from app.services.profile_builder import UserProfileBuilder
from app.services.profile_compactor import UserProfileCompactor
from app.services.prompt_assembler import PromptAssembler
from app.services.settings import LLMSettings, load_llm_settings
from app.services.token_cost import ProviderTokenCostEstimator
from app.services.viability_evaluator import PersonalizationViabilityEvaluator
from app.storage.memory_store import InMemoryUserMapStore


def test_negative_feedback_increases_depth_preference():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()

    analysis = analyzer.analyze(
        user_message="That was not what I meant. I wanted a deeper explanation.",
        assistant_response="Short answer.",
    )

    updater.apply_analysis(user_map, analysis)

    assert "pref_depth" in user_map.nodes
    assert user_map.nodes["pref_depth"].weight > 0
    assert user_map.nodes["signal_satisfaction"].weight < 0
    assert user_map.nodes["signal_confusion"].weight < 0


def test_brief_unsuccessful_answer_pushes_toward_depth():
    analyzer = InteractionAnalyzer()

    analysis = analyzer.analyze(
        user_message="That missed the point. I needed more detail.",
        assistant_response="Here is a short answer.",
    )

    assert analysis.depth_preference_delta > analysis.concision_preference_delta
    assert any("too brief" in note for note in analysis.notes)


def test_context_builder_outputs_active_nodes():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    builder = PersonalizationContextBuilder()

    analysis = analyzer.analyze(
        user_message="Thank you, that was helpful. I like thorough technical answers about LLMs.",
        assistant_response="Detailed answer.",
    )

    updater.apply_analysis(user_map, analysis)
    context = builder.build_context(user_map)

    assert "Personalization context" in context
    assert "Response style:" in context or "Likely recurring topics:" in context


def test_context_builder_describes_confusion_as_reduced_signal():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    builder = PersonalizationContextBuilder()

    analysis = analyzer.analyze(
        user_message="That was not what I meant. I wanted a deeper explanation.",
        assistant_response="Short answer.",
    )

    updater.apply_analysis(user_map, analysis)
    context = builder.build_context(user_map)

    assert "recent confusion trend reduced" in context


def test_updater_persists_engagement_signal():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()

    analysis = analyzer.analyze(
        user_message="Thank you, that was helpful.",
        assistant_response="Detailed answer.",
    )

    updater.apply_analysis(user_map, analysis)

    assert user_map.nodes["signal_engagement"].weight > 0


def test_store_persists_maps_to_disk():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        user_map = store.get_or_create("persisted-user")
        analyzer = InteractionAnalyzer()
        updater = UserMapUpdater()

        analysis = analyzer.analyze(
            user_message="Thanks, the detailed explanation was helpful.",
            assistant_response="Detailed answer about LLM memory.",
        )

        updater.apply_analysis(user_map, analysis)
        store.save(user_map)
        store.record_interaction(
            user_id="persisted-user",
            user_message="Thanks, the detailed explanation was helpful.",
            assistant_response="Detailed answer about LLM memory.",
            explicit_feedback=None,
            analysis=analysis,
        )

        reloaded_store = InMemoryUserMapStore(storage_path=storage_path)

        assert reloaded_store.get_or_create("persisted-user").nodes["signal_engagement"].weight > 0
        assert len(reloaded_store.get_recent_interactions("persisted-user")) == 1
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_profile_builder_groups_traits_by_role():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    profile_builder = UserProfileBuilder()

    analysis = analyzer.analyze(
        user_message="Thank you, that was helpful. I like thorough technical answers about LLMs.",
        assistant_response="Detailed answer about LLM memory.",
    )

    updater.apply_analysis(user_map, analysis)
    profile = profile_builder.build_profile(user_map)

    assert profile.user_id == "test-user"
    assert profile.communication_preferences
    assert profile.short_term_topic_interests


def test_frequent_topics_decay_slower_than_infrequent_topics():
    user_map = UserMap(user_id="test-user")
    updater = UserMapUpdater()
    now = datetime.utcnow()

    user_map.nodes["topic_short_frequent"] = UserNode(
        id="topic_short_frequent",
        label="Short-term interest in frequent topic",
        type=NodeType.TOPIC_INTEREST,
        weight=0.8,
        confidence=0.8,
        evidence_count=8,
        last_updated=now - timedelta(days=10),
    )
    user_map.nodes["topic_short_infrequent"] = UserNode(
        id="topic_short_infrequent",
        label="Short-term interest in infrequent topic",
        type=NodeType.TOPIC_INTEREST,
        weight=0.8,
        confidence=0.8,
        evidence_count=1,
        last_updated=now - timedelta(days=10),
    )

    updater.apply_decay(user_map, now=now)

    assert user_map.nodes["topic_short_frequent"].weight > user_map.nodes["topic_short_infrequent"].weight


def test_context_builder_uses_compact_summary_without_numeric_overhead():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    builder = PersonalizationContextBuilder()

    analysis = analyzer.analyze(
        user_message="Thank you, that was helpful. I like thorough technical answers about LLMs and memory.",
        assistant_response="Detailed answer about LLM memory.",
    )

    updater.apply_analysis(user_map, analysis)
    context = builder.build_context(user_map)

    assert "weight=" not in context
    assert "confidence=" not in context
    assert len(context.splitlines()) <= 4


def test_compactor_promotes_repeated_short_term_topics_to_long_term_topics():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    compactor = UserProfileCompactor()

    for _ in range(3):
        analysis = analyzer.analyze(
            user_message="I keep asking about LLM memory and personalization.",
            assistant_response="Detailed answer about LLM memory.",
        )
        updater.apply_analysis(user_map, analysis)

    result = compactor.compact(user_map)

    assert result["promoted_topics"] >= 1
    assert "topic_long_ai_personalization" in user_map.nodes


def test_compactor_prunes_stale_weak_nodes():
    user_map = UserMap(user_id="test-user")
    compactor = UserProfileCompactor()
    stale_time = datetime.utcnow() - timedelta(days=30)
    user_map.nodes["signal_stale"] = UserNode(
        id="signal_stale",
        label="Stale weak signal",
        type=NodeType.EMOTIONAL_SIGNAL,
        weight=0.01,
        confidence=0.01,
        evidence_count=1,
        last_updated=stale_time,
    )

    result = compactor.compact(user_map)

    assert result["pruned_nodes"] == 1
    assert "signal_stale" not in user_map.nodes


def test_prompt_assembler_includes_recent_evidence_when_profile_is_weak():
    user_map = UserMap(user_id="test-user")
    assembler = PromptAssembler(PersonalizationContextBuilder())
    analyzer = InteractionAnalyzer()
    analysis = analyzer.analyze(
        user_message="I wanted more detail on memory.",
        assistant_response="Short answer.",
    )
    recent_interactions = [
        {
            "user_message": "I wanted more detail on memory.",
            "assistant_response": "Short answer.",
            "explicit_feedback": None,
            "analysis_json": analysis.model_dump_json(),
            "created_at": datetime.utcnow().isoformat(),
        }
    ]

    prompt_payload = assembler.assemble(user_map, recent_interactions)

    assert prompt_payload.supporting_evidence
    assert "Use the recent evidence only to disambiguate weak profile signals." in prompt_payload.prompt_instructions


def test_prompt_assembler_skips_evidence_when_profile_is_strong():
    user_map = UserMap(user_id="test-user")
    analyzer = InteractionAnalyzer()
    updater = UserMapUpdater()
    assembler = PromptAssembler(PersonalizationContextBuilder())

    for _ in range(4):
        analysis = analyzer.analyze(
            user_message="Thank you, I prefer thorough technical answers about LLM memory.",
            assistant_response="Detailed answer about LLM memory.",
        )
        updater.apply_analysis(user_map, analysis)

    prompt_payload = assembler.assemble(user_map, recent_interactions=[])

    assert not prompt_payload.supporting_evidence


def test_store_lists_user_ids_for_compaction():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        store.save(UserMap(user_id="a-user"))
        store.save(UserMap(user_id="b-user"))

        assert store.list_user_ids() == ["a-user", "b-user"]
    finally:
        if storage_path.exists():
            storage_path.unlink()


class FakeLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[str, str]] = []
        self.stream_calls: list[tuple[str, str]] = []

    def generate_response(self, user_message: str, prompt_assembly):
        self.calls.append((user_message, prompt_assembly.prompt_instructions))
        return self.response_text

    def stream_response(self, user_message: str, prompt_assembly):
        self.stream_calls.append((user_message, prompt_assembly.prompt_instructions))
        for token in ["Here ", "is ", "streamed."]:
            yield token


def test_chat_service_uses_prompt_assembly_and_persists_interaction():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        analyzer = InteractionAnalyzer()
        updater = UserMapUpdater()
        context_builder = PersonalizationContextBuilder()
        prompt_builder = PromptAssembler(context_builder)
        profile_compactor = UserProfileCompactor()
        fake_llm = FakeLLMClient("Here is a thorough answer about LLM memory.")
        chat_service = PersonalizedChatService(
            store=store,
            analyzer=analyzer,
            updater=updater,
            prompt_assembler=prompt_builder,
            context_builder=context_builder,
            profile_compactor=profile_compactor,
            llm_client=fake_llm,
        )

        response = chat_service.generate_reply(
            user_id="chat-user",
            user_message="Can you explain LLM memory in more detail?",
        )

        assert response.user_id == "chat-user"
        assert "LLM memory" in response.assistant_response
        assert fake_llm.calls
        assert store.get_recent_interactions("chat-user")
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_chat_service_updates_profile_after_generated_response():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        analyzer = InteractionAnalyzer()
        updater = UserMapUpdater()
        context_builder = PersonalizationContextBuilder()
        prompt_builder = PromptAssembler(context_builder)
        profile_compactor = UserProfileCompactor()
        fake_llm = FakeLLMClient("Here is a detailed technical explanation about LLM memory and user maps.")
        chat_service = PersonalizedChatService(
            store=store,
            analyzer=analyzer,
            updater=updater,
            prompt_assembler=prompt_builder,
            context_builder=context_builder,
            profile_compactor=profile_compactor,
            llm_client=fake_llm,
        )

        chat_service.generate_reply(
            user_id="chat-user",
            user_message="I want a technical explanation about LLM memory.",
        )

        user_map = store.get_or_create("chat-user")

        assert "pref_depth" in user_map.nodes
        assert any(node_id.startswith("topic_short_") for node_id in user_map.nodes)
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_chat_service_applies_explicit_feedback_to_persisted_interaction():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        analyzer = InteractionAnalyzer()
        updater = UserMapUpdater()
        context_builder = PersonalizationContextBuilder()
        prompt_builder = PromptAssembler(context_builder)
        profile_compactor = UserProfileCompactor()
        fake_llm = FakeLLMClient("Here is a short answer.")
        chat_service = PersonalizedChatService(
            store=store,
            analyzer=analyzer,
            updater=updater,
            prompt_assembler=prompt_builder,
            context_builder=context_builder,
            profile_compactor=profile_compactor,
            llm_client=fake_llm,
        )

        chat_service.generate_reply(
            user_id="feedback-user",
            user_message="Explain LLM memory.",
        )
        interaction = store.get_recent_interactions("feedback-user", limit=1)[0]

        feedback_response = chat_service.apply_feedback(
            interaction_id=int(interaction["interaction_id"]),
            explicit_feedback="That was too shallow. I wanted more detail.",
        )

        assert feedback_response.analysis.depth_preference_delta > 0
        refreshed_interaction = store.get_interaction(int(interaction["interaction_id"]))
        assert refreshed_interaction is not None
        assert refreshed_interaction["explicit_feedback"] == "That was too shallow. I wanted more detail."
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_load_llm_settings_uses_environment_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    settings = load_llm_settings()

    assert settings.provider == "openai"
    assert settings.model == "gpt-4o-mini"
    assert settings.api_key is None


def test_load_llm_settings_supports_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")

    settings = load_llm_settings()

    assert settings.provider == "anthropic"
    assert settings.api_key == "test-key"
    assert settings.model == "claude-test"
    assert settings.base_url == "https://example.invalid"


def test_prompt_assembly_is_cached_by_profile_version():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        user_map = store.get_or_create("cache-user")
        prompt_builder = PromptAssembler(PersonalizationContextBuilder())
        prompt = prompt_builder.assemble(user_map, recent_interactions=[])
        store.save_cached_prompt_assembly(prompt)

        cached_prompt = store.get_cached_prompt_assembly("cache-user", user_map.updated_at.isoformat())

        assert cached_prompt is not None
        assert cached_prompt.profile_version == user_map.updated_at.isoformat()
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_chat_service_streams_tokens_and_persists_final_response():
    storage_path = Path("data") / f"test_user_maps_{uuid4().hex}.db"

    try:
        store = InMemoryUserMapStore(storage_path=storage_path)
        analyzer = InteractionAnalyzer()
        updater = UserMapUpdater()
        context_builder = PersonalizationContextBuilder()
        prompt_builder = PromptAssembler(context_builder)
        profile_compactor = UserProfileCompactor()
        fake_llm = FakeLLMClient("unused")
        chat_service = PersonalizedChatService(
            store=store,
            analyzer=analyzer,
            updater=updater,
            prompt_assembler=prompt_builder,
            context_builder=context_builder,
            profile_compactor=profile_compactor,
            llm_client=fake_llm,
        )

        events = list(chat_service.stream_reply("stream-user", "Stream a reply."))

        assert any('"event": "token"' in event for event in events)
        assert any('"event": "done"' in event for event in events)
        assert store.get_recent_interactions("stream-user", limit=1)[0]["assistant_response"] == "Here is streamed."
    finally:
        if storage_path.exists():
            storage_path.unlink()


def test_viability_evaluator_reports_personalization_lift():
    evaluator = PersonalizationViabilityEvaluator(
        analyzer=InteractionAnalyzer(),
        updater=UserMapUpdater(),
        profile_compactor=UserProfileCompactor(),
    )

    result = evaluator.run_experiment(rounds_per_user=4)

    assert result.supports_viability
    assert result.personalized_mean_alignment > result.baseline_mean_alignment
    assert result.personalized_mean_satisfaction > result.baseline_mean_satisfaction
    assert result.cost_benefit.worth_it
    assert result.report_metadata.app_version == __version__


def test_viability_evaluator_returns_per_scenario_metrics():
    evaluator = PersonalizationViabilityEvaluator(
        analyzer=InteractionAnalyzer(),
        updater=UserMapUpdater(),
        profile_compactor=UserProfileCompactor(),
    )

    result = evaluator.run_experiment(rounds_per_user=3)

    assert result.scenario_results
    assert all(scenario.alignment_lift >= 0 for scenario in result.scenario_results)
    assert any(scenario.scenario_source == "human_labeled" for scenario in result.scenario_results)
    assert result.exported_artifacts


def test_viability_evaluator_exports_json_and_csv_reports():
    report_dir = Path("data") / f"reports_{uuid4().hex}"

    try:
        evaluator = PersonalizationViabilityEvaluator(
            analyzer=InteractionAnalyzer(),
            updater=UserMapUpdater(),
            profile_compactor=UserProfileCompactor(),
            report_dir=report_dir,
        )

        result = evaluator.run_experiment(rounds_per_user=2)

        artifact_formats = {artifact.format for artifact in result.exported_artifacts}
        assert artifact_formats == {"json", "csv"}
        assert all(Path(artifact.path).exists() for artifact in result.exported_artifacts)
        csv_artifact = next(artifact for artifact in result.exported_artifacts if artifact.format == "csv")
        with Path(csv_artifact.path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            first_row = next(reader)
        assert first_row["app_version"] == __version__
        assert first_row["llm_provider"] == result.report_metadata.llm_provider
    finally:
        if report_dir.exists():
            for child in report_dir.iterdir():
                child.unlink()
            report_dir.rmdir()


def test_viability_evaluator_reports_token_cost_tradeoff():
    evaluator = PersonalizationViabilityEvaluator(
        analyzer=InteractionAnalyzer(),
        updater=UserMapUpdater(),
        profile_compactor=UserProfileCompactor(),
    )

    result = evaluator.run_experiment(rounds_per_user=3)

    assert result.personalized_mean_token_cost > result.baseline_mean_token_cost
    assert result.cost_benefit.mean_token_cost_delta > 0
    assert result.cost_benefit.alignment_lift_per_token_delta > 0


def test_provider_token_cost_estimator_uses_provider_specific_pricing():
    openai_estimator = ProviderTokenCostEstimator(
        LLMSettings(provider="openai", model="gpt-4o-mini")
    )
    anthropic_estimator = ProviderTokenCostEstimator(
        LLMSettings(provider="anthropic", model="claude-3-5-sonnet-latest")
    )

    openai_cost = openai_estimator.estimate_chat_cost(
        system_prompt="You are helpful.",
        user_message="Explain persistent memory.",
        assistant_response="Here is a detailed explanation of persistent memory.",
    )
    anthropic_cost = anthropic_estimator.estimate_chat_cost(
        system_prompt="You are helpful.",
        user_message="Explain persistent memory.",
        assistant_response="Here is a detailed explanation of persistent memory.",
    )

    assert openai_cost.total_tokens > 0
    assert anthropic_cost.total_tokens > 0
    assert openai_cost.estimated_cost_usd != anthropic_cost.estimated_cost_usd


def test_cli_viability_benchmark_writes_reports(capsys):
    report_dir = Path("data") / f"cli_reports_{uuid4().hex}"

    try:
        exit_code = cli_main([
            "viability-benchmark",
            "--rounds-per-user",
            "2",
            "--report-dir",
            str(report_dir),
        ])

        captured = capsys.readouterr()

        assert exit_code == 0
        assert "viability supports=" in captured.out
        assert report_dir.exists()
        assert any(report_dir.iterdir())
    finally:
        if report_dir.exists():
            for child in report_dir.iterdir():
                child.unlink()
            report_dir.rmdir()
