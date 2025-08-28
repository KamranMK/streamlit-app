import logging
import instructor
import boto3
from typing import List, Type
from pydantic import BaseModel

class AnthropicBedrockChatCompletions:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create(
        self,
        modelId: str,
        max_tokens: int,
        system_message: str,
        messages: List,
        response_model: BaseModel | Type[BaseModel],
        dump: bool = True
    ):
        """
        Send prompt and acquire the chat completions response.

        :param modelId str: the llm model we are going to use
        :param max tokens: token limitation
        :param system_message: system_message for api call
        :param messages: list of client messages
        :param target_key: key of target field
        :param response_model: tools we are using
        """
        bedrock_client = boto3.client('bedrock-runtime')
        client = instructor.from_bedrock(bedrock_client)
        try:
            resp = client.chat.completions.create(
                modelId=modelId,
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
                response_model=response_model
            )
            self.logger.info("Prompt is successful")
            # return the pydantic object without introducing breaking change
            if not dump:
                return resp
            return resp.model_dump()

        except Exception as e:
            self.logger.error(f"Could not complete prompt due to: {str(e)}")
            raise RuntimeError(f"Could not complete prompt due to: {str(e)}")
