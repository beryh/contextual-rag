import boto3
from typing import Dict
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class EvalutationDynamoDBWriter:
    def __init__(self, aws_region: str, aws_profile: str, table_name: str):
        self.table_name = table_name
        self.dynamodb = boto3.Session(
            region_name=aws_region, 
            profile_name=aws_profile
        ).client('dynamodb')
        self._create_table_if_not_exists(self.table_name)

    def save_question(self, question: Dict) -> None:
        item = {
            "question_id": {'S': question['question_id']},
            "timestamp": {'S': datetime.now(timezone.utc).isoformat()},
            "question": {'S': question['question']},
            "ground_truth": {'S': question['ground_truth']},
            "question_type": {'S': question['question_type']},
            "context": {'S': question['context']}
        }

        self._put_item(item)
        logger.info(f"Saved question data ({question['question_id']})")

    def save_result(self, contextual: bool, hybrid: bool, question: Dict, result: Dict) -> None:
        item = self._build_item(contextual, hybrid, question, result)
        self._put_item(item)
        logger.info(f"Saved result for question {question}")

    def _create_table_if_not_exists(self, table_name: str):
        if table_name not in self.dynamodb.list_tables()['TableNames']:
            logger.info(f"Creating Table: {table_name}")
            self.dynamodb.create_table(
                TableName=table_name,
                AttributeDefinitions=[
                    {'AttributeName': 'uuid', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                KeySchema=[
                    {'AttributeName': 'uuid', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 10,
                    'WriteCapacityUnits': 10
                }
            )

    def _put_item(self, item: Dict) -> None:
        try:
            logger.info(f"Putting item to {self.table_name}")
            self.dynamodb.put_item(
                TableName=self.table_name,
                Item=item
            )
        except Exception as e:
            logger.error(f"Error saving to DynamoDB: {e}")
            raise

    @staticmethod
    def _build_item(contextual: bool, hybrid: bool, question: str, result: Dict, evaluation: Dict) -> Dict:
        return {
            'question_id': {'S': question},
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
            'answer': {'S': result['answer']},
            'elapsedTime': {'S': str(result['elapsed_time'])},
            'eveluation': {
                "N": evaluation['context_precision']
            }
        }