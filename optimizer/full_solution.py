import logging
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.core.generator import Generator
from adalflow.core.component import Component
from adalflow.core.base_data_class import DataClass
from adalflow.optim.parameter import Parameter
from adalflow.optim.types import ParameterType
from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.trainer.trainer import Trainer
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from adalflow.optim.text_grad.ops import BackwardEngine
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# Your Bedrock client imports
import instructor
import boto3
from typing import Type
from pydantic import BaseModel

# Example structured output model for text classification
@dataclass
class EmailClassificationOutput(DataClass):
    """Structured output for email classification task"""
    category: str = field(
        metadata={"desc": "The email category: 'work', 'personal', 'spam', or 'promotional'"}
    )
    reasoning: str = field(
        metadata={"desc": "Step-by-step reasoning for the classification decision"}
    )
    confidence: float = field(
        metadata={"desc": "Confidence score between 0 and 1 for the classification"}
    )
    
    __input_fields__ = []
    __output_fields__ = ["category", "reasoning", "confidence"]

@dataclass 
class EmailDataSample(DataClass):
    """Data sample for email classification"""
    email_content: str = field(metadata={"desc": "The email content to classify"})
    true_category: str = field(metadata={"desc": "The ground truth category"})
    id: str = field(metadata={"desc": "Unique identifier for the sample"})

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
        """Send prompt and acquire the chat completions response."""
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

class OfflineCompatibleModelClient(ModelClient):
    """Bedrock model client configured for offline optimization."""
    
    def __init__(self, model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"):
        super().__init__()
        self.model_id = model_id
        self.logger = logging.getLogger("OfflineCompatibleModelClient")
        self.bedrock_client = AnthropicBedrockChatCompletions()
        self.logger.info("OfflineCompatibleModelClient initialized")
        
    def convert_inputs_to_api_kwargs(
        self, 
        input: Optional[Any] = None, 
        model_kwargs: Dict = {}, 
        model_type: ModelType = ModelType.UNDEFINED
    ) -> Dict:
        """Convert AdalFlow standard inputs to API-specific format."""
        
        # Use EmailClassificationOutput as default response model
        api_kwargs = {
            "modelId": model_kwargs.get("model", self.model_id),
            "max_tokens": model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": str(input)}],
            "system_message": "You are an expert email classifier. Analyze the email content and provide a structured classification.",
            "response_model": EmailClassificationOutput,
            "dump": True
        }
        
        # Override with any provided model_kwargs
        if "system_message" in model_kwargs:
            api_kwargs["system_message"] = model_kwargs["system_message"]
        if "response_model" in model_kwargs:
            api_kwargs["response_model"] = model_kwargs["response_model"]
        if "dump" in model_kwargs:
            api_kwargs["dump"] = model_kwargs["dump"]
            
        return api_kwargs
    
    def call(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Make the actual API call."""
        try:
            response = self.bedrock_client.create(**api_kwargs)
            return GeneratorOutput(data=response, error=None, usage=None, raw_response=response)
        except Exception as e:
            self.logger.error(f"Error calling Bedrock: {e}", exc_info=True)
            return GeneratorOutput(data=None, error=str(e), usage=None, raw_response=None)
    
    async def acall(self, api_kwargs: Dict = {}, model_type: ModelType = ModelType.UNDEFINED):
        """Async version of the call method."""
        return self.call(api_kwargs, model_type)

class EmailClassificationPipeline(Component):
    """
    Email classification task pipeline with structured output.
    This is the main task component that will be optimized.
    """
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict):
        super().__init__()
        
        # Create optimizable parameters
        self.system_prompt = Parameter(
            data="""You are an expert email classifier. Analyze the email content carefully and classify it into one of these categories:
- work: Professional emails, meetings, business communications
- personal: Personal messages, family, friends
- spam: Unwanted promotional emails, suspicious content
- promotional: Legitimate marketing emails, newsletters

Provide your reasoning step-by-step and assign a confidence score.""",
            role_desc="The system prompt that guides the email classification task",
            param_type=ParameterType.PROMPT,
            requires_opt=True
        )
        
        # Template for the email classification
        template = r"""<SYS>
{{system_prompt}}
</SYS>

Email to classify:
{{email_content}}

Please analyze this email and provide:
1. Category (work/personal/spam/promotional)
2. Your reasoning
3. Confidence score (0-1)
"""
        
        # Create the generator with structured output
        self.generator = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={"system_prompt": self.system_prompt},
            output_processors=None  # Let structured output handle parsing
        )
    
    def call(self, email_content: str, id: str = None) -> GeneratorOutput:
        """Process a single email classification request."""
        return self.generator.call(prompt_kwargs={"email_content": email_content})

class EmailClassificationOptimizer(AdalComponent):
    """
    AdalComponent for optimizing email classification with offline support.
    This handles the optimization pipeline while working in offline environments.
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        teacher_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        # Create the task pipeline
        task = EmailClassificationPipeline(model_client, model_kwargs)
        
        # Create evaluation function
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # Create loss function with explicit backward engine configuration
        # This is the key to making it work offline - we provide explicit config
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Exact match between predicted and ground truth category",
            backward_engine=None,  # Will be configured later in configure_backward_engine
            model_client=None,     # Will be configured later
            model_kwargs=None      # Will be configured later
        )
        
        # Initialize the parent with all required components
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
        )
        
        self.logger = logging.getLogger("EmailClassificationOptimizer")
        self.logger.info("EmailClassificationOptimizer initialized")
    
    def configure_backward_engine(self, *args, **kwargs):
        """
        Configure the backward engine for optimization.
        This method is called by the Trainer and handles offline initialization.
        """
        self.logger.info("Configuring backward engine for offline usage...")
        
        try:
            # Create the backward engine with the same model client
            backward_engine = BackwardEngine(
                model_client=self.task.generator.model_client,
                model_kwargs=self.backward_engine_model_config
            )
            
            # Set the backward engine on the loss function
            self.loss_fn.set_backward_engine(
                backward_engine=backward_engine,
                model_client=self.task.generator.model_client,
                model_kwargs=self.backward_engine_model_config
            )
            
            self.logger.info("Backward engine configured successfully")
            
        except Exception as e:
            self.logger.error(f"Error configuring backward engine: {e}")
            # In offline mode, we can still proceed with demo optimization only
            self.logger.warning("Continuing without backward engine - only demo optimization available")
    
    def prepare_task(self, sample: EmailDataSample) -> Tuple[Callable, Dict]:
        """Prepare a single task sample for processing."""
        return self.task.call, {"email_content": sample.email_content, "id": sample.id}
    
    def prepare_eval(self, sample: EmailDataSample, y_pred: GeneratorOutput) -> Tuple[Callable, Dict]:
        """Prepare evaluation for a single sample."""
        # Extract the predicted category from structured output
        pred_category = None
        if y_pred and y_pred.data and hasattr(y_pred.data, 'category'):
            pred_category = y_pred.data.category
        elif y_pred and y_pred.data and isinstance(y_pred.data, dict):
            pred_category = y_pred.data.get('category')
        
        return self.eval_fn, {"y": pred_category, "y_gt": sample.true_category}
    
    def prepare_loss(self, sample: EmailDataSample, y_pred: Parameter) -> Tuple[Callable, Dict]:
        """Prepare loss computation for a single sample."""
        # Create ground truth parameter
        y_gt = Parameter(
            data=sample.true_category,
            eval_input=sample.true_category,
            requires_opt=False,
            role_desc="Ground truth email category"
        )
        
        # Set evaluation input for the prediction
        if y_pred.full_response and y_pred.full_response.data:
            if hasattr(y_pred.full_response.data, 'category'):
                y_pred.eval_input = y_pred.full_response.data.category
            elif isinstance(y_pred.full_response.data, dict):
                y_pred.eval_input = y_pred.full_response.data.get('category')
        
        return self.loss_fn.forward, {"kwargs": {"y": y_pred, "y_gt": y_gt}}

def create_offline_training_setup():
    """
    Create a complete offline-compatible training setup for email classification.
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OfflineTrainingSetup")
    
    logger.info("Setting up offline-compatible email classification optimization...")
    
    # Model configurations - using the same model for all components in offline mode
    model_config = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    # Create model client
    model_client = OfflineCompatibleModelClient()
    
    # Create the AdalComponent for optimization
    adal_component = EmailClassificationOptimizer(
        model_client=model_client,
        model_kwargs=model_config,
        backward_engine_model_config=model_config,  # Same model for backward engine
        teacher_model_config=model_config,          # Same model for teacher
        text_optimizer_model_config=model_config,   # Same model for text optimizer
    )
    
    # Create trainer with conservative settings for offline mode
    trainer = Trainer(
        adaltask=adal_component,
        strategy="random_sampling",  # Use random sampling instead of constrained
        max_steps=5,  # Fewer steps for offline testing
        num_workers=1,  # Single worker to avoid connection issues
        optimization_order="demo_and_prompt",  # Optimize both demos and prompts
        raw_shots=1,      # Minimal few-shot examples
        bootstrap_shots=1, # Minimal bootstrap examples
    )
    
    logger.info("Offline training setup completed successfully!")
    
    return adal_component, trainer, model_client

# Example usage and testing
if __name__ == "__main__":
    
    # Create sample data for testing
    train_samples = [
        EmailDataSample(
            email_content="Meeting scheduled for tomorrow at 2 PM to discuss quarterly results.",
            true_category="work",
            id="1"
        ),
        EmailDataSample(
            email_content="Hey! Want to grab dinner tonight? Let me know!",
            true_category="personal", 
            id="2"
        ),
        EmailDataSample(
            email_content="CONGRATULATIONS! You've won $1,000,000! Click here now!",
            true_category="spam",
            id="3"
        ),
        EmailDataSample(
            email_content="New products now available with 20% discount. Shop now!",
            true_category="promotional",
            id="4"
        ),
    ]
    
    val_samples = train_samples[:2]  # Use subset for validation
    test_samples = train_samples[2:]  # Use subset for testing
    
    try:
        # Set up the offline training
        adal_component, trainer, model_client = create_offline_training_setup()
        
        print("Testing basic functionality...")
        
        # Test individual task execution
        sample = train_samples[0]
        result = adal_component.run_one_task_sample(sample)
        print(f"Task result: {result}")
        
        # Test evaluation
        pred_result = adal_component.task.call(sample.email_content, sample.id)
        eval_result = adal_component.prepare_eval(sample, pred_result)
        print(f"Evaluation result: {eval_result}")
        
        print("\nOffline setup is working! Ready for optimization.")
        print("To run full optimization, use:")
        print("trainer.fit(train_dataset=train_samples, val_dataset=val_samples, test_dataset=test_samples)")
        
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
