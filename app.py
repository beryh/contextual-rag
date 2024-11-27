import json
import logging
from datetime import datetime
from time import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import boto3
import requests
import botocore.config
from requests_aws4auth import AWS4Auth
from opensearchpy import OpenSearch, RequestsHttpConnection

import os
from dotenv import load_dotenv

from question import QuestionLoader
from config import Config

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    model_id: str
    embed_model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    system_prompt: str

    @classmethod
    def from_env(cls):
        return cls(
            model_id=os.getenv('CLAUDE_MODEL_ID'),
            embed_model_id=os.getenv('EMBED_MODEL_ID'),
            max_tokens=int(os.getenv('CLAUDE_MAX_TOKENS')),
            temperature=float(os.getenv('CLAUDE_TEMPERATURE')),
            top_p=float(os.getenv('CLAUDE_TOP_P')),
            system_prompt=os.getenv('CLAUDE_SYSTEM_PROMPT')
        )

@dataclass
class Question:
    id: str
    question: str

class RAGService:
    def __init__(self, config: Config):
        self.region = config.aws.region
        self.aws_profile = config.aws.profile
        self.bedrock_client = self._init_bedrock_client()
        self.opensearch_client = self._init_opensearch_client()
        self.rerank_api_url = config.reranker.api_url
        
    def _init_bedrock_client(self):
        retry_config = botocore.config.Config(
            retries={"max_attempts": 10, "mode": "standard"}
        )
        session = boto3.Session(
            region_name=self.region, 
            profile_name=self.aws_profile
        )
        return session.client("bedrock-runtime", config=retry_config, region_name=self.region)

    def _init_opensearch_client(self) -> OpenSearch:
        session = boto3.Session(region_name=self.region, profile_name='ml')
        os_client = session.client('opensearch')
        
        domain_name = [domain['DomainName'] for domain in os_client.list_domain_names().get('DomainNames')][0]
        host = os_client.describe_domain(DomainName=domain_name)['DomainStatus']['Endpoint']
        
        credentials = session.get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            self.region,
            "es",
            session_token=credentials.token
        )
        
        return OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=300
        )

    def get_embedding(self, text: str, model_id: str) -> List[float]:
        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({"inputText": text})
            )
            return json.loads(response['body'].read())['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def search_documents(self, question: str, index_name: str, 
                        use_hybrid: bool, embed_model_id: str) -> List[Dict]:
        embedding = self.get_embedding(question, embed_model_id)
        
        if use_hybrid:
            knn_results = self._search_by_knn(embedding, index_name, top_n=40)
            bm25_results = self._search_by_bm25(question, index_name, top_n=40)
            return self._rank_fusion(question, knn_results, bm25_results)
        else:
            return self._search_by_knn(embedding, index_name, top_n=20)

    def _search_by_knn(self, vector: List[float], index_name: str, top_n: int = 80) -> List[Dict]:
        query = {
            "size": top_n,
            "_source": ["content", "metadata"],
            "query": {
                "knn": {
                    "content_embedding": {
                        "vector": vector,
                        "k": top_n
                    }
                }
            }
        }

        try:
            response = self.opensearch_client.search(index=index_name, body=query)
            return [self._format_search_result(hit, 'knn') 
                   for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"KNN search error: {e}")
            return []

    def _search_by_bm25(self, query_text: str, index_name: str, top_n: int = 80) -> List[Dict]:
        query = {
            "size": top_n,
            "_source": ["content", "metadata"],
            "query": {
                "match": {
                    "content": {
                        "query": query_text,
                        "operator": "or"
                    }
                }
            }
        }

        try:
            response = self.opensearch_client.search(index=index_name, body=query)
            return [self._format_search_result(hit, 'bm25') 
                   for hit in response['hits']['hits']]
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []

    def _rerank_documents(self, question, documents, top_k=20):
        payload = {
            "documents": documents,
            "query": question,
            "rank_fields": ["content"],
            "top_n": top_k
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.rerank_api_url, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            return result 
        else:
            print(f"Error: API failed (status code: {response.status_code})")
            print(f"response: {response.text}")
            return None


    def _rank_fusion(self, question, knn_results, bm25_results, hybrid_score_filter=40, final_reranked_results=20, knn_weight=0.6):
        bm25_weight = 1 - knn_weight

        def _normalize_and_weight_score(results, weight):
            if not results:
                return results
            min_score = min(r['score'] for r in results)
            max_score = max(r['score'] for r in results)
            score_range = max_score - min_score
            if score_range == 0:
                return results
            for r in results:
                r['normalized_score'] = ((r['score'] - min_score) / score_range) * weight
            return results

        knn_results = _normalize_and_weight_score(knn_results, knn_weight)
        bm25_results = _normalize_and_weight_score(bm25_results, bm25_weight)
        
        # Combine results and calculate hybrid score
        combined_results = {}
        for result in knn_results + bm25_results:
            chunk_id = result['metadata'].get('chunk_id', result['content']) 
            if chunk_id not in combined_results:
                combined_results[chunk_id] = result.copy()
                combined_results[chunk_id]['hybrid_score'] = result.get('normalized_score', 0)
                combined_results[chunk_id]['search_methods'] = [result['search_method']]
            else:
                combined_results[chunk_id]['hybrid_score'] += result.get('normalized_score', 0)
                if result['search_method'] not in combined_results[chunk_id]['search_methods']:
                    combined_results[chunk_id]['search_methods'].append(result['search_method'])

        # Convert back to list and sort by hybrid score
        results_list = list(combined_results.values())
        results_list.sort(key=lambda x: x['hybrid_score'], reverse=True)
        hybrid_results = results_list[:hybrid_score_filter]

        # Prepare documents for reranking
        documents_for_rerank = [
            {"content": doc['content'], "metadata": doc['metadata']} for doc in hybrid_results
        ]

        # Rerank the documents -> return ranked indices
        reranked_results = self._rerank_documents(question, documents_for_rerank, final_reranked_results)

        # Prepare final results
        if reranked_results and isinstance(reranked_results, dict) and 'results' in reranked_results:
            final_results = []
            for reranked_doc in reranked_results['results']:
                if isinstance(reranked_doc, dict) and 'index' in reranked_doc and 'relevance_score' in reranked_doc:
                    index = reranked_doc['index']
                    if 0 <= index < len(hybrid_results):
                        original_doc = hybrid_results[index]
                        final_doc = {
                            "content": original_doc["content"],
                            'metadata': original_doc['metadata'],
                            'score': reranked_doc['relevance_score'], 
                            'hybrid_score': original_doc['hybrid_score'],
                            'search_methods': original_doc['search_methods']
                        }   
                        final_results.append(final_doc)
                else:
                    logger.warning(f"Unexpected reranked document format: {reranked_doc}")

            final_results.sort(key=lambda x: x['score'], reverse=True)

        else:
            logger.warning("Reranking failed or returned unexpected format. Using hybrid results.")
            final_results = [{
                "content": doc["content"],
                'metadata': doc['metadata'],
                'score': doc['hybrid_score'],
                'hybrid_score': doc['hybrid_score'],
                'search_methods': doc['search_methods']
            } for doc in hybrid_results[:final_reranked_results]]
            
        return final_results
    
    @staticmethod
    def _format_search_result(hit: Dict, search_method: str) -> Dict:
        return {
            "content": hit['_source']["content"],
            "score": hit['_score'],
            "metadata": hit['_source']['metadata'],
            "search_method": search_method
        }
    
    def process(self, document: str, question: str, chunk_size: int, 
           use_contextual: bool, use_hybrid: bool, model_config: ModelConfig):

        # get timestamp
        start_dt = datetime.now()

        # embed question
        embedded_question = self.get_embedding(question, model_config.embed_model_id)
        index_name = f"aws_{'contextual_' if use_contextual else ''}{document}_{chunk_size}"
        logger.info(f"Using index: {index_name}")

        if use_hybrid:
            search_result_knn = self._search_by_knn(embedded_question, index_name, top_n=40)
            search_result_bm25 = self._search_by_bm25(question, index_name, top_n=40)        
            search_result = self._rank_fusion(question, search_result_knn, search_result_bm25)
        else:
            search_result = self._search_by_knn(embedded_question, index_name, top_n=20)


        additional_info = ""
        for result in search_result:
            additional_info += f"- {result['content']}\n\n"

        messages = [{
            'role': 'user',
            'content': [{'text': f"{question}\n\nAdditional Information:\n{additional_info}"}]
        }]

        # try:
        for attempt in range(10):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_config.model_id,
                    messages=messages,
                    system=[{'text': model_config.system_prompt}],
                    inferenceConfig={
                        'maxTokens': model_config.max_tokens,
                        'temperature': model_config.temperature,
                        'topP': model_config.top_p
                    }
                )
                break
            except botocore.errorfactory.ThrottlingException as e:
                print(f"({attempt}) An error occurred during answer generation: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"Some Exception occurred: {e}")
                break

        # get elapsed time
        elapsed_time = str((datetime.now() - start_dt).total_seconds())
        
        result = {
            'timestamp': start_dt.isoformat(),
            'question': question,
            # 'answer': response['output']['message']['content'][0]['text'],
            # 'retrieved_contexts': [ctx['content'] for ctx in search_result],
            'usage': response['usage'],
            'latency': response['metrics']['latencyMs'],
            'elapsed_time': elapsed_time
        }
        print(result)

        return result
    

class ResultProcessor:
    def __init__(self, config: Config):
        self.region = config.aws.region
        self.aws_profile = config.aws.profile
        self.table_name = config.dynamodb.table
        self.dynamodb = boto3.Session(
            region_name=self.region, 
            profile_name=self.aws_profile
        ).client('dynamodb')

    def save_result(self, contextual: bool, hybrid: bool, 
                   question: Question, result: Dict) -> None:
        try:
            self.dynamodb.put_item(
                TableName='contextual_rag_result',
                Item={
                    'question_id': {'S': question.id},
                    'timestamp': {'S': result['timestamp']},
                    'useContextual': {'BOOL': contextual},
                    'useHybrid': {'BOOL': hybrid},
                    'token': {
                        'M': {
                            "input": {"N": str(result['usage']['inputTokens'])},
                            "output": {"N": str(result['usage']['outputTokens'])},
                            "total": {"N": str(result['usage']['totalTokens'])}
                        }
                    },
                    'elapsedTime': {'S': result['elapsed_time']}
                }
            )
        except Exception as e:
            logger.error(f"Error saving to DynamoDB: {e}")
            raise

def main():
    # Load configuration
    config = Config.load()
    
    # Initialize services
    rag_service = RAGService(config)
    result_processor = ResultProcessor(config)

    # Load questions
    try:
        question_loader = QuestionLoader()
        questions = question_loader.load_questions()
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return

    # Process questions
    for question in questions:
        logger.info(f"Processing question {question.id}")
        for contextual in [True, False]:
            for hybrid in [True, False]:
                try:
                    result = rag_service.process(
                        document="bedrock",
                        question=question.question,
                        chunk_size=1000,
                        use_contextual=contextual,
                        use_hybrid=hybrid,
                        model_config=config.model
                    )
                    result_processor.save_result(contextual, hybrid, question, result)
                    time.sleep(60)  # Rate limiting
                except Exception as e:
                    logger.error(f"Error processing question {question.id}: {e}")
                    continue

if __name__ == "__main__":
    main()