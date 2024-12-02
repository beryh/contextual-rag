import boto3
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class DynamoDBWriter:
    def __init__(self, aws_region: str, aws_profile: str, table_name: str):
        self.table_name = table_name
        self.dynamodb = boto3.Session(
            region_name=aws_region, 
            profile_name=aws_profile
        ).client('dynamodb')
        self._create_table_if_not_exists(self.table_name)

    def save_result(self, contextual: bool, hybrid: bool, question: str, result: Dict) -> None:
        item = self._build_item(contextual, hybrid, question, result)
        self._put_item(item)
        logger.info(f"Saved result for question {question}")

    def _create_table_if_not_exists(self, table_name: str):
        if table_name not in self.dynamodb.list_tables()['TableNames']:
            self.dynamodb.create_table(
                TableName=table_name,
                AttributeDefinitions=[
                    {'AttributeName': 'question_id', 'AttributeType': 'S'},
                    {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                ],
                KeySchema=[
                    {'AttributeName': 'question_id', 'KeyType': 'HASH'},
                    {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                ],
                ProvisionedThroughput={
                    'ReadCapacityUnits': 10,
                    'WriteCapacityUnits': 10
                }
            )

    def _put_item(self, item: Dict) -> None:
        try:
            self.dynamodb.put_item(
                TableName=self.table_name,
                Item=item
            )
        except Exception as e:
            logger.error(f"Error saving to DynamoDB: {e}")
            raise

    @staticmethod
    def _build_item(contextual: bool, hybrid: bool, question: str, result: Dict) -> Dict:
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
            'elapsedTime': {'S': str(result['elapsed_time'])}
        }