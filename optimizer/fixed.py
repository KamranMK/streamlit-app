import logging
from typing import Dict, Any, Optional
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.core.component import Component
from adalflow.core.generator import Generator

# Your Bedrock client imports
from your_bedrock_module import EmailLabel, DEFAULT_MODEL_ID  # Replace with actual module name

# Your AnthropicBedrockChatCompletions class
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
        system_message: str | List,
        messages: List,
        response_model: BaseModel | Type[BaseModel],
        dump: bool = True
    ):
        """
        Send prompt and acquire the chat completions response.
        """
        bedrock_client = boto3.client('bedrock-runtime')
        client = instructor.from_bedrock(bedrock_client)
        
        # convert system message to list in case its not
        if not isinstance(system_message, List):
            system_message = [{'role': 'system', 'content': system_message}]
        
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

class SimpleModelClient(ModelClient):
    """
    A simplified version of the Bedrock model client for debugging purposes.
    This extends ModelClient from adalflow to be compatible with Generator.
    """
    
    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        # Initialize the parent ModelClient
        super().__init__()
        self.model_id = model_id
        self.logger = logging.getLogger("SimpleModelClient")
        self.logger.info("SimpleModelClient initialized")
        self.bedrock_client = AnthropicBedrockChatCompletions()
        
    def convert_inputs_to_api_kwargs(
        self, 
        input: Optional[Any] = None, 
        model_kwargs: Dict = {}, 
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Dict:
        """
        Convert AdalFlow standard inputs to API-specific format.
        This method is required by the Generator component.
        """
        self.logger.info(f"convert_inputs_to_api_kwargs called with input: {input}, model_kwargs: {model_kwargs}")
        
        # Convert the input to the format expected by your Bedrock client
        api_kwargs = {
            "modelId": model_kwargs.get("model", self.model_id),
            "max_tokens": model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": str(input)}],
            "system_message": "You are a helpful, accurate AI assistant.",
            "response_model": EmailLabel,
            "dump": True
        }
        
        # Override system_message if provided in model_kwargs
        if "system_message" in model_kwargs:
            api_kwargs["system_message"] = model_kwargs["system_message"]
        
        # Override response_model if provided in model_kwargs
        if "response_model" in model_kwargs:
            api_kwargs["response_model"] = model_kwargs["response_model"]
        
        # Override dump if provided in model_kwargs
        if "dump" in model_kwargs:
            api_kwargs["dump"] = model_kwargs["dump"]
        
        self.logger.info(f"Generated api_kwargs: {api_kwargs}")
        return api_kwargs
    
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        Make the actual API call with the sync client.
        This method is required by the Generator component.
        """
        self.logger.info(f"call method invoked with api_kwargs: {api_kwargs}")
        
        try:
            # Make direct call to Bedrock using your client
            self.logger.info("Making direct call to Bedrock")
            
            response = self.bedrock_client.create(**api_kwargs)
            self.logger.info(f"Got response: {response}")
            
            # Return GeneratorOutput with proper structure
            return GeneratorOutput(data=response, error=None, usage=None, raw_response=response)
            
        except Exception as e:
            self.logger.error(f"Error calling Bedrock: {e}", exc_info=True)
            return GeneratorOutput(data=None, error=str(e), usage=None, raw_response=None)
    
    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """
        Async version of the call method.
        For now, just call the sync version.
        """
        self.logger.info("acall method invoked (using sync call)")
        return self.call(api_kwargs, model_type)

# Test code
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting debug execution")
    
    try:
        # Create a simple EmailLabel model for testing (replace with your actual model)
        class TestEmailLabel(BaseModel):
            label: str
            confidence: float
        
        # Test the simple client directly
        client = SimpleModelClient()
        logger.info("Created SimpleModelClient")
        
        # Test with AdalFlow Generator
        logger.info("Testing with AdalFlow Generator")
        anthropic_llm = Generator(
            model_client=client,
            model_kwargs={
                "max_tokens": 1024,
                "response_model": TestEmailLabel,  # Use your actual EmailLabel here
                "system_message": "You are a helpful email classifier."
            }
        )
        logger.info("Generator created successfully")
        
        # This should now work
        generator_response = anthropic_llm(prompt_kwargs={"input_str": "I have a problem with my wardrobe."})
        logger.info(f"Generator response: {generator_response}")
        
    except Exception as e:
        logger.error(f"Debug execution failed: {e}", exc_info=True)
