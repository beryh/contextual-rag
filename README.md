# Contextual RAG System

이 프로젝트는 Amazon Bedrock을 활용한 Contextual RAG(Retrieval Augmented Generation) 시스템입니다.

## 주요 기능

- Amazon Bedrock Foundation Model을 활용한 텍스트 생성
- 임베딩 기반 유사도 검색(KNN) 및 키워드 기반 검색(BM25) 하이브리드 검색
- Rank Fusion과 Cross-encoder Reranking을 통한 검색 결과 최적화
- DynamoDB를 활용한 결과 저장 및 관리

## 시스템 구성

- `BedrockService`: AWS Bedrock 서비스와의 상호작용 담당
- `RerankerService`: 검색 결과 재순위화 처리
- `Config`: 환경 설정 관리

## 설치 및 설정

1. 필요한 환경 변수 설정 (.env 파일)
```
AWS Configuration
AWS_REGION: AWS 리전 설정 (기본값: ap-northeast-2)
AWS_PROFILE: AWS 프로파일 설정 (기본값: default)

Bedrock Configuration  
BEDROCK_MODEL_ID: Bedrock 모델 ID (예: anthropic.claude-3-5-sonnet-20240620-v1:0)
BEDROCK_RETRIES: API 재시도 횟수 (기본값: 10)
EMBED_MODEL_ID: 임베딩 모델 ID (예: amazon.titan-embed-text-v2:0)

OpenSearch Configuration
OPENSEARCH_PREFIX: OpenSearch 인덱스 접두사 
OPENSEARCH_DOMAIN_NAME: OpenSearch 도메인 이름
OPENSEARCH_DOCUMENT_NAME: 문서 인덱스 이름
OPENSEARCH_USER: OpenSearch 사용자 이름
OPENSEARCH_PASSWORD: OpenSearch 비밀번호

DynamoDB Configuration
DYNAMODB_TABLE_NAME: DynamoDB 테이블 이름 (기본값: contextual_rag_result)

Reranker Configuration
RERANKER_AWS_REGION: Reranker 서비스 리전 (기본값: us-west-2)
RERANKER_AWS_PROFILE: Reranker AWS 프로파일 (기본값: default)
RERANKER_MODEL_ID: Reranker 모델 ID (예: amazon.rerank-v1:0)

Rank Fusion Configuration
RERANK_TOP_K: 최종 재순위화 결과 수 (기본값: 20)
HYBRID_SCORE_FILTER: 하이브리드 검색 결과 필터링 수 (기본값: 40)
FINAL_RERANKED_RESULTS: 최종 재순위화 결과 수 (기본값: 20)
KNN_WEIGHT: KNN 검색 가중치 (기본값: 0.6)

Model Configuration
MAX_TOKENS: 최대 토큰 수 (기본값: 4096)
TEMPERATURE: 생성 텍스트의 무작위성 (기본값: 0)
TOP_P: 상위 확률 샘플링 임계값 (기본값: 0.7)

Application Configuration
CHUNK_SIZE: 문서 청크 크기 (기본값: 1000)
RATE_LIMIT_DELAY: API 요청 간 지연 시간(초) (기본값: 60)
```
2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

