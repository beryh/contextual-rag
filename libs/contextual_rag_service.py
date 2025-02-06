from datetime import datetime

from libs.bedrock_service import BedrockService
from libs.opensearch_service import OpensearchService
from libs.reranker import RerankerService

import logging
logger = logging.getLogger(__name__)

class ContextualRAGService:
    def __init__(self, bedrock_service: BedrockService, opensearch_service: OpensearchService, reranker_service: RerankerService):
        self.bedrock_service = bedrock_service
        self.opensearch_service = opensearch_service
        self.reranker_service = reranker_service

    def do(self, question: str, index_name: str, use_hybrid: bool, search_limit: int):
        # get timestamp
        start_dt = datetime.now()

        # search
        embedding = self.bedrock_service.embedding(question)
        
        if use_hybrid:
            knn_results = self.opensearch_service.search_by_knn(embedding, index_name, search_limit)
            bm25_results = self.opensearch_service.search_by_bm25(question, index_name, search_limit)
            search_results = self.reranker_service.rank_fusion(question, knn_results, bm25_results, final_reranked_results=search_limit)
        else:
            search_results = self.opensearch_service.search_by_knn(embedding, index_name, search_limit)

        docs = ""
        for result in search_results:
            docs += f"- {result['content']}\n\n"

        messages = [{
            'role': 'user',
            'content': [{'text': f"{question}\n\nAdditional Information:\n{docs}"}]
        }]

        processing_time = (datetime.now() - start_dt).microseconds // 1_000
        system_prompt = "You are a helpful AI assistant that provides accurate and concise information about Amazon Bedrock."

        response = self.bedrock_service.converse(
            messages=messages,
            system_prompt=system_prompt
        )

        result = {
            'timestamp': start_dt.isoformat(),
            'question': question,
            'answer': response['output']['message']['content'][0]['text'],
            'retrieved_contexts': response['output']['message']['content'],
            'usage': response['usage'],
            'latency': response['metrics']['latencyMs'],
            'elapsed_time': processing_time + response['metrics']['latencyMs']
        }

        return result
    
    def do(self, question: str, document_name: str, chunk_size: str, use_hybrid: bool, use_contextual: bool, search_limit: int):
        # build actual index name
        index_name = f"{'contextual_' if use_contextual else ''}{document_name}_{chunk_size}"

        return self.do(question=question, index_name=index_name, use_hybrid=use_hybrid, search_limit=search_limit)

