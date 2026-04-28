from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse, InteractionRequest, InteractionResult, PromptAssembly, UserMap, UserProfile, ViabilityExperimentResult
from app.storage.memory_store import store
from app.services.analyzer import InteractionAnalyzer
from app.services.chat_service import PersonalizedChatService
from app.services.map_updater import UserMapUpdater
from app.services.context_builder import PersonalizationContextBuilder
from app.services.profile_builder import UserProfileBuilder
from app.services.profile_compactor import UserProfileCompactor
from app.services.llm_client import build_llm_client
from app.services.prompt_assembler import PromptAssembler
from app.services.settings import load_llm_settings
from app.services.viability_evaluator import PersonalizationViabilityEvaluator

app = FastAPI(
    title="Holistic Temporal User Map",
    description="Starter API for building structured, evolving personalization maps from user interactions.",
    version="0.1.0",
)

analyzer = InteractionAnalyzer()
updater = UserMapUpdater()
context_builder = PersonalizationContextBuilder()
profile_builder = UserProfileBuilder()
profile_compactor = UserProfileCompactor()
prompt_assembler = PromptAssembler(context_builder)
viability_evaluator = PersonalizationViabilityEvaluator(
    analyzer=analyzer,
    updater=updater,
    profile_compactor=profile_compactor,
)


def _build_chat_service() -> PersonalizedChatService:
    llm_settings = load_llm_settings()
    return PersonalizedChatService(
        store=store,
        analyzer=analyzer,
        updater=updater,
        prompt_assembler=prompt_assembler,
        context_builder=context_builder,
        profile_compactor=profile_compactor,
        llm_client=build_llm_client(settings=llm_settings),
    )


@app.get("/")
def root():
    return {
        "project": "Holistic Temporal User Map",
        "status": "running",
        "docs": "/docs",
    }


@app.post("/api/interactions", response_model=InteractionResult)
def process_interaction(payload: InteractionRequest):
    user_map = store.get_or_create(payload.user_id)

    analysis = analyzer.analyze(
        user_message=payload.user_message,
        assistant_response=payload.assistant_response,
        explicit_feedback=payload.explicit_feedback,
    )

    updated_nodes = updater.apply_analysis(user_map, analysis)
    profile_compactor.compact(user_map)
    store.save(user_map)
    store.record_interaction(
        user_id=payload.user_id,
        user_message=payload.user_message,
        assistant_response=payload.assistant_response,
        explicit_feedback=payload.explicit_feedback,
        analysis=analysis,
    )

    personalization_context = context_builder.build_context(user_map)

    return InteractionResult(
        user_id=payload.user_id,
        analysis=analysis,
        updated_nodes=updated_nodes,
        personalization_context=personalization_context,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    try:
        chat_service = _build_chat_service()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return chat_service.generate_reply(
        user_id=payload.user_id,
        user_message=payload.message,
    )


@app.post("/api/chat/stream")
def stream_chat(payload: ChatRequest):
    try:
        chat_service = _build_chat_service()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        chat_service.stream_reply(
            user_id=payload.user_id,
            user_message=payload.message,
        ),
        media_type="application/x-ndjson",
    )


@app.post("/api/interactions/{interaction_id}/feedback", response_model=FeedbackResponse)
def apply_feedback(interaction_id: int, payload: FeedbackRequest):
    try:
        chat_service = _build_chat_service()
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        return chat_service.apply_feedback(
            interaction_id=interaction_id,
            explicit_feedback=payload.explicit_feedback,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/users/{user_id}/map", response_model=UserMap)
def get_user_map(user_id: str):
    user_map = store.get_or_create(user_id)
    if updater.apply_decay(user_map):
        store.save(user_map)
    return user_map


@app.get("/api/users/{user_id}/context")
def get_personalization_context(user_id: str):
    user_map = store.get_or_create(user_id)
    if updater.apply_decay(user_map):
        profile_compactor.compact(user_map)
        store.save(user_map)
    return {
        "user_id": user_id,
        "context": context_builder.build_context(user_map),
    }


@app.get("/api/users/{user_id}/profile", response_model=UserProfile)
def get_user_profile(user_id: str):
    user_map = store.get_or_create(user_id)
    if updater.apply_decay(user_map):
        profile_compactor.compact(user_map)
        store.save(user_map)
    return profile_builder.build_profile(user_map)


@app.get("/api/users/{user_id}/history")
def get_user_history(user_id: str, limit: int = 20):
    return {
        "user_id": user_id,
        "interactions": store.get_recent_interactions(user_id, limit=limit),
    }


@app.get("/api/users/{user_id}/prompt-context", response_model=PromptAssembly)
def get_prompt_context(user_id: str):
    user_map = store.get_or_create(user_id)
    if updater.apply_decay(user_map):
        profile_compactor.compact(user_map)
        store.save(user_map)
    recent_interactions = store.get_recent_interactions(user_id, limit=3)
    return prompt_assembler.assemble(user_map, recent_interactions)


@app.post("/api/admin/compact-profiles")
def compact_profiles():
    results = []
    for user_id in store.list_user_ids():
        user_map = store.get_or_create(user_id)
        updater.apply_decay(user_map)
        compacted = profile_compactor.compact(user_map)
        store.save(user_map)
        results.append({"user_id": user_id, **compacted})

    return {
        "processed_profiles": len(results),
        "results": results,
    }


@app.get("/api/evaluations/viability", response_model=ViabilityExperimentResult)
def run_viability_evaluation(rounds_per_user: int = 4):
    return viability_evaluator.run_experiment(rounds_per_user=rounds_per_user)
