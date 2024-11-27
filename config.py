from dataclasses import dataclass
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AWSConfig:
    region: str = os.getenv('AWS_REGION', 'ap-northeast-2')
    profile: str = os.getenv('AWS_PROFILE', 'ml')

@dataclass
class ModelConfig:
    model_id: str = os.getenv('CLAUDE_MODEL_ID')
    max_tokens: int = int(os.getenv('CLAUDE_MAX_TOKENS', 4096))
    temperature: float = float(os.getenv('CLAUDE_TEMPERATURE', 0))
    top_p: float = float(os.getenv('CLAUDE_TOP_P', 0.7))
    system_prompt: str = os.getenv('CLAUDE_SYSTEM_PROMPT')
    embed_model_id: str = os.getenv('EMBED_MODEL_ID')

@dataclass
class OpenSearchConfig:
    index: str = os.getenv('OPENSEARCH_INDEX', 'bedrock-docs')

@dataclass
class DynamoDBConfig:
    table: str = os.getenv('DYNAMODB_TABLE', 'contextual_rag_result')

@dataclass
class RerankerConfig:
    api_url: str = os.getenv('RERANK_API_URL')
    top_k: int = int(os.getenv('RERANK_TOP_K', 20))

@dataclass
class RankFusionConfig:
    hybrid_score_filter: int = int(os.getenv('HYBRID_SCORE_FILTER', 40))
    final_reranked_results: int = int(os.getenv('FINAL_RERANKED_RESULTS', 20))
    knn_weight: float = float(os.getenv('KNN_WEIGHT', 0.6))

@dataclass
class AppConfig:
    chunk_size: int = int(os.getenv('CHUNK_SIZE', 1000))
    rate_limit_delay: int = int(os.getenv('RATE_LIMIT_DELAY', 60))

class Config:
    def __init__(self):
        self.aws = AWSConfig()
        self.model = ModelConfig()
        self.opensearch = OpenSearchConfig()
        self.dynamodb = DynamoDBConfig()
        self.reranker = RerankerConfig()
        self.rank_fusion = RankFusionConfig()
        self.app = AppConfig()

    @classmethod
    def load(cls) -> 'Config':
        return cls()