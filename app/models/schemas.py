from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class NodeType(str, Enum):
    PREFERENCE = "preference"
    COMMUNICATION_STYLE = "communication_style"
    COGNITIVE_STYLE = "cognitive_style"
    EMOTIONAL_SIGNAL = "emotional_signal"
    TOPIC_INTEREST = "topic_interest"
    TRUST_SIGNAL = "trust_signal"


class UserNode(BaseModel):
    id: str
    label: str
    type: NodeType
    weight: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    evidence_count: int = 0
    metadata: Dict[str, str] = Field(default_factory=dict)


class UserEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    evidence_count: int = 0


class UserMap(BaseModel):
    user_id: str
    nodes: Dict[str, UserNode] = Field(default_factory=dict)
    edges: List[UserEdge] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class InteractionRequest(BaseModel):
    user_id: str
    user_message: str
    assistant_response: str
    explicit_feedback: Optional[str] = None


class InteractionAnalysis(BaseModel):
    satisfaction_score: float
    confusion_score: float
    depth_preference_delta: float
    concision_preference_delta: float
    emotional_engagement_delta: float
    detected_topics: List[str]
    notes: List[str]


class InteractionResult(BaseModel):
    user_id: str
    analysis: InteractionAnalysis
    updated_nodes: List[UserNode]
    personalization_context: str


class ProfileTrait(BaseModel):
    id: str
    label: str
    weight: float
    confidence: float
    evidence_count: int


class UserProfile(BaseModel):
    user_id: str
    communication_preferences: List[ProfileTrait] = Field(default_factory=list)
    cognitive_preferences: List[ProfileTrait] = Field(default_factory=list)
    short_term_topic_interests: List[ProfileTrait] = Field(default_factory=list)
    long_term_topic_interests: List[ProfileTrait] = Field(default_factory=list)
    emotional_signals: List[ProfileTrait] = Field(default_factory=list)
    trust_signals: List[ProfileTrait] = Field(default_factory=list)
    updated_at: datetime


class PromptAssembly(BaseModel):
    user_id: str
    profile_version: str
    profile_confidence: float
    context: str
    supporting_evidence: List[str] = Field(default_factory=list)
    prompt_instructions: str


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    user_message: str
    assistant_response: str
    prompt_assembly: PromptAssembly
    analysis: InteractionAnalysis
    updated_nodes: List[UserNode]
    personalization_context: str


class FeedbackRequest(BaseModel):
    explicit_feedback: str


class FeedbackResponse(BaseModel):
    interaction_id: int
    user_id: str
    explicit_feedback: str
    analysis: InteractionAnalysis
    updated_nodes: List[UserNode]
    personalization_context: str


class ChatStreamChunk(BaseModel):
    event: str
    user_id: str
    delta: str = ""
    done: bool = False
    interaction_id: int | None = None
    personalization_context: str | None = None


class ViabilityScenarioResult(BaseModel):
    scenario_name: str
    scenario_source: str
    baseline_mean_alignment: float
    personalized_mean_alignment: float
    baseline_mean_satisfaction: float
    personalized_mean_satisfaction: float
    baseline_mean_token_cost: float
    personalized_mean_token_cost: float
    alignment_lift: float
    satisfaction_lift: float
    token_cost_delta: float
    alignment_lift_per_token_delta: float


class EvaluationArtifact(BaseModel):
    format: str
    path: str


class EvaluationReportMetadata(BaseModel):
    generated_at: datetime
    git_revision: Optional[str] = None
    app_version: str
    llm_provider: str
    llm_model: str
    tokenizer_name: str


class CostBenefitEvaluation(BaseModel):
    baseline_mean_token_cost: float
    personalized_mean_token_cost: float
    mean_token_cost_delta: float
    alignment_lift_per_token_delta: float
    satisfaction_lift_per_token_delta: float
    worth_it_threshold: float
    worth_it: bool


class ViabilityExperimentResult(BaseModel):
    hypothesis: str
    primary_metric: str
    secondary_metric: str
    report_metadata: EvaluationReportMetadata
    rounds_per_user: int
    scenario_count: int
    baseline_mean_alignment: float
    personalized_mean_alignment: float
    baseline_mean_satisfaction: float
    personalized_mean_satisfaction: float
    baseline_mean_token_cost: float
    personalized_mean_token_cost: float
    mean_alignment_lift: float
    mean_satisfaction_lift: float
    cost_benefit: CostBenefitEvaluation
    supports_viability: bool
    methodology: List[str]
    scenario_results: List[ViabilityScenarioResult]
    exported_artifacts: List[EvaluationArtifact] = Field(default_factory=list)
