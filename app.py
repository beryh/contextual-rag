import logging
import time
from dotenv import load_dotenv

from question import QuestionLoader
from config import Config

from libs.bedrock_service import BedrockService
from libs.opensearch_service import OpensearchService
from libs.reranker import RerankerService
from libs.dynamodb_writer import DynamoDBWriter

from libs.contextual_rag_service import ContextualRAGService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def main():
    # Load configuration
    config = Config.load()
    
    # Initialize services
    bedrock_service = BedrockService(config.aws.region, config.aws.profile, config.bedrock.retries, config.bedrock.embed_model_id, config.bedrock.model_id, config.model.max_tokens, config.model.temperature, config.model.top_p)
    opensearch_service = OpensearchService(config.aws.region, config.aws.profile, config.opensearch.prefix, config.opensearch.domain_name, config.opensearch.document_name, config.opensearch.user, config.opensearch.password)
    reranker_service = RerankerService(config.reranker.aws_region, config.reranker.aws_profile, config.reranker.reranker_model_id, config.bedrock.retries)
    
    rag_service = ContextualRAGService(bedrock_service, opensearch_service, reranker_service)
    ddb_writer = DynamoDBWriter(config.aws.region, config.aws.profile, config.dynamodb.table_name)

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
            # get index name
            for hybrid in [True, False]:
                try:
                    result = rag_service.do(
                        question=question.question,
                        document_name=config.opensearch.document_name,
                        chunk_size=config.app.chunk_size,
                        use_hybrid=hybrid,
                        use_contextual=contextual,
                        search_limit=40
                    )
                except Exception as e:
                    logger.error(f"Error processing question {question.id}: {e}")
                    continue

                # TODO: Evaluation with RAGAS

                ddb_writer.save_result(contextual, hybrid, question.id, result)
                time.sleep(config.app.rate_limit_delay)  # Rate limiting

if __name__ == "__main__":
    main()