import json
from collections.abc import Iterator

from app.models.schemas import ChatResponse, ChatStreamChunk, FeedbackResponse, PromptAssembly, UserMap
from app.services.analyzer import InteractionAnalyzer
from app.services.context_builder import PersonalizationContextBuilder
from app.services.llm_client import LLMClient
from app.services.map_updater import UserMapUpdater
from app.services.profile_compactor import UserProfileCompactor
from app.services.prompt_assembler import PromptAssembler
from app.storage.memory_store import InMemoryUserMapStore


class PersonalizedChatService:
    def __init__(
        self,
        store: InMemoryUserMapStore,
        analyzer: InteractionAnalyzer,
        updater: UserMapUpdater,
        prompt_assembler: PromptAssembler,
        context_builder: PersonalizationContextBuilder,
        profile_compactor: UserProfileCompactor,
        llm_client: LLMClient,
    ):
        self._store = store
        self._analyzer = analyzer
        self._updater = updater
        self._prompt_assembler = prompt_assembler
        self._context_builder = context_builder
        self._profile_compactor = profile_compactor
        self._llm_client = llm_client

    def generate_reply(self, user_id: str, user_message: str) -> ChatResponse:
        user_map, prompt_assembly = self._prepare_prompt_assembly(user_id)
        assistant_response = self._llm_client.generate_response(user_message, prompt_assembly)

        return self._finalize_reply(
            user_map=user_map,
            user_message=user_message,
            assistant_response=assistant_response,
            prompt_assembly=prompt_assembly,
        )

    def stream_reply(self, user_id: str, user_message: str) -> Iterator[str]:
        user_map, prompt_assembly = self._prepare_prompt_assembly(user_id)
        collected_chunks: list[str] = []

        for chunk in self._llm_client.stream_response(user_message, prompt_assembly):
            collected_chunks.append(chunk)
            yield self._serialize_stream_chunk(
                ChatStreamChunk(
                    event="token",
                    user_id=user_id,
                    delta=chunk,
                )
            )

        assistant_response = "".join(collected_chunks).strip()
        response = self._finalize_reply(
            user_map=user_map,
            user_message=user_message,
            assistant_response=assistant_response,
            prompt_assembly=prompt_assembly,
        )
        interaction = self._store.get_recent_interactions(user_id, limit=1)[0]
        yield self._serialize_stream_chunk(
            ChatStreamChunk(
                event="done",
                user_id=user_id,
                done=True,
                interaction_id=int(interaction["interaction_id"]),
                personalization_context=response.personalization_context,
            )
        )

    def _prepare_prompt_assembly(self, user_id: str) -> tuple[UserMap, PromptAssembly]:
        user_map = self._store.get_or_create(user_id)
        if self._updater.apply_decay(user_map):
            self._profile_compactor.compact(user_map)
            self._store.save(user_map)
            self._store.invalidate_prompt_cache(user_id)

        profile_version = user_map.updated_at.isoformat()
        cached_prompt = self._store.get_cached_prompt_assembly(user_id, profile_version)
        if cached_prompt is not None:
            return user_map, cached_prompt

        recent_interactions = self._store.get_recent_interactions(user_id, limit=3)
        prompt_assembly = self._prompt_assembler.assemble(user_map, recent_interactions)
        self._store.save_cached_prompt_assembly(prompt_assembly)
        return user_map, prompt_assembly

    def _finalize_reply(
        self,
        user_map: UserMap,
        user_message: str,
        assistant_response: str,
        prompt_assembly: PromptAssembly,
    ) -> ChatResponse:
        if not assistant_response:
            raise ValueError("LLM returned an empty response.")

        analysis = self._analyzer.analyze(
            user_message=user_message,
            assistant_response=assistant_response,
            explicit_feedback=None,
        )
        updated_nodes = self._updater.apply_analysis(user_map, analysis)
        self._profile_compactor.compact(user_map)
        self._store.save(user_map)
        self._store.invalidate_prompt_cache(user_map.user_id)
        self._store.record_interaction(
            user_id=user_map.user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            explicit_feedback=None,
            analysis=analysis,
        )

        personalization_context = self._context_builder.build_context(user_map)

        return ChatResponse(
            user_id=user_map.user_id,
            user_message=user_message,
            assistant_response=assistant_response,
            prompt_assembly=prompt_assembly,
            analysis=analysis,
            updated_nodes=updated_nodes,
            personalization_context=personalization_context,
        )

    def apply_feedback(self, interaction_id: int, explicit_feedback: str) -> FeedbackResponse:
        interaction = self._store.get_interaction(interaction_id)
        if interaction is None:
            raise ValueError(f"Interaction {interaction_id} was not found.")

        user_id = str(interaction["user_id"])
        user_message = str(interaction["user_message"])
        assistant_response = str(interaction["assistant_response"])
        user_map = self._store.get_or_create(user_id)

        analysis = self._analyzer.analyze(
            user_message=user_message,
            assistant_response=assistant_response,
            explicit_feedback=explicit_feedback,
        )
        updated_nodes = self._updater.apply_analysis(user_map, analysis)
        self._profile_compactor.compact(user_map)
        self._store.save(user_map)
        self._store.invalidate_prompt_cache(user_id)
        self._store.update_interaction_feedback(interaction_id, explicit_feedback, analysis)

        personalization_context = self._context_builder.build_context(user_map)

        return FeedbackResponse(
            interaction_id=interaction_id,
            user_id=user_id,
            explicit_feedback=explicit_feedback,
            analysis=analysis,
            updated_nodes=updated_nodes,
            personalization_context=personalization_context,
        )

    def _serialize_stream_chunk(self, chunk: ChatStreamChunk) -> str:
        return json.dumps(chunk.model_dump(mode="json")) + "\n"