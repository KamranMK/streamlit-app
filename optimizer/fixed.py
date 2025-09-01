import logging
from typing import Dict, Any, Optional
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput

# Your Bedrock client imports
# from your_bedrock_module import EmailLabel, DEFAULT_MODEL_ID  # Replace with actual module name

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

# Create a simple wrapper that mimics Generator without optimization features
class SimpleGenerator:
    """
    A simplified Generator that works offline by avoiding optimization components.
    This directly uses the ModelClient without triggering AdalFlow's optimization layer.
    """
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict = None):
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.logger = logging.getLogger("SimpleGenerator")
        self.logger.info("SimpleGenerator initialized (offline mode)")
    
    def __call__(self, prompt_kwargs: Dict) -> GeneratorOutput:
        """
        Process the input using the model client directly.
        """
        self.logger.info(f"SimpleGenerator called with prompt_kwargs: {prompt_kwargs}")
        
        try:
            # Extract input from prompt_kwargs
            input_str = prompt_kwargs.get("input_str", "")
            
            # Convert to API kwargs using the model client
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=input_str,
                model_kwargs=self.model_kwargs,
                model_type=ModelType.LLM
            )
            
            # Make the call
            response = self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            
            self.logger.info("SimpleGenerator response generated successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"SimpleGenerator error: {e}", exc_info=True)
            return GeneratorOutput(data=None, error=str(e), usage=None, raw_response=None)
    
    def call(self, prompt_kwargs: Dict) -> GeneratorOutput:
        """Alias for __call__ method"""
        return self.__call__(prompt_kwargs)

class SimpleModelClient(ModelClient):
    """
    A simplified version of the Bedrock model client for offline debugging purposes.
    This extends ModelClient from adalflow to be compatible with SimpleGenerator.
    """
    
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
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
        
        # Create a default EmailLabel for testing
        class DefaultEmailLabel(BaseModel):
            category: str
            confidence: float
            reasoning: str
        
        # Convert the input to the format expected by your Bedrock client
        api_kwargs = {
            "modelId": model_kwargs.get("model", self.model_id),
            "max_tokens": model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": str(input)}],
            "system_message": "You are a helpful, accurate AI assistant.",
            "response_model": model_kwargs.get("response_model", DefaultEmailLabel),
            "dump": True
        }
        
        # Override system_message if provided in model_kwargs
        if "system_message" in model_kwargs:
            api_kwargs["system_message"] = model_kwargs["system_message"]
        
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

# Test code - Using SimpleGenerator instead of Generator to avoid optimization issues
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting offline debug execution")
    
    try:
        # Create your EmailLabel model for testing
        class EmailLabel(BaseModel):
            category: str = "general"
            confidence: float = 0.9
            reasoning: str = "Default reasoning"
        
        # Test the simple client directly
        client = SimpleModelClient()
        logger.info("Created SimpleModelClient")
        
        # Use SimpleGenerator instead of Generator to avoid optimization components
        logger.info("Testing with SimpleGenerator (offline-compatible)")
        simple_llm = SimpleGenerator(
            model_client=client,
            model_kwargs={
                "max_tokens": 1024,
                "response_model": EmailLabel,
                "system_message": "You are a helpful email classifier. Classify the following message.",
                "model": "anthropic.claude-3-sonnet-20240229-v1:0"
            }
        )
        logger.info("SimpleGenerator created successfully")
        
        # This should work without hanging
        test_messages = [
            "I have a problem with my wardrobe.",
            "Please schedule a meeting for tomorrow.",
            "Thank you for your help with the project."
        ]
        
        for i, message in enumerate(test_messages):
            logger.info(f"Processing message {i+1}: {message}")
            response = simple_llm(prompt_kwargs={"input_str": message})
            logger.info(f"Response {i+1}: {response}")
            
            if response.data:
                logger.info(f"Successful response data: {response.data}")
            else:
                logger.warning(f"Error in response: {response.error}")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Debug execution failed: {e}", exc_info=True)
