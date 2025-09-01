I have this code from adalflow.core.model_client import ModelClient
from adalflow.core.component import Component
from adalflow.core.generator import Generator# Extend ModelClient from adalflow instead of creating a standalone class
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
        
    def __call__(self, prompt_kwargs):
        self.logger.info(f"SimpleModelClient called with: {prompt_kwargs}")
        
        # Extract the input string - this is what's likely coming from AdalFlow
        input_str = prompt_kwargs.get("input_str", "")
        self.logger.info(f"Input string: {input_str}")
        
        # Convert the input string to messages format
        messages = [{"role": "user", "content": input_str}]
        system_message = [{"role": "system", "content": "You are a helpful, accurate AI assistant."}]
        
        try:
            # Make direct call to Bedrock
            self.logger.info("Making direct call to Bedrock")
            response = self.bedrock_client.create(
                modelId=self.model_id,
                max_tokens=1024,
                system_message=system_message,
                messages=messages,
                response_model=EmailLabel,
                dump=True
            )
            self.logger.info(f"Got response: {response}")
            return response
        except Exception as e:
            self.logger.error(f"Error calling Bedrock: {e}", exc_info=True)
            return f"Error: {str(e)}"when I ran this, just simple model client, all workslogger.info("Starting debug execution")
    
try:
    # Test the simple client directly without AdalFlow
    client = SimpleModelClient()
    logger.info("Created SimpleModelClient")
    
    response = client({"input_str": "I have a problem with my wardrobe."})
    logger.info(f"Response received: {response}")
except Exception as e:
    logger.error(f"Debug execution failed: {e}", exc_info=True)but when i ran this# Create the generator with your custom client
anthropic_llm = Generator(
    model_client=SimpleModelClient(),
    # model_kwargs={"max_tokens": 1024}  # Set any additional model parameters
)its stuck hereINFO:SimpleModelClient:SimpleModelClient initialized
INFO:adalflow.optim.grad_component:EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs.
INFO:adalflow.core.generator:Generator Generator initialized.just keeps saying generator initialized, what is wrong?I am using adalflow 1.1.2
