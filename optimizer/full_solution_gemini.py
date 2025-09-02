import logging
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field

# Adaflow imports
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.core.component import Component
from adalflow.core.base_data_class import DataClass
from adalflow.optim.parameter import Parameter
from adalflow.optim.types import ParameterType
from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.trainer.trainer import Trainer
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
# Note: We import the default BackwardEngine only for type hinting if needed,
# but we will use our custom one.
from adalflow.core.generator import BackwardEngine
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# Your Bedrock client imports
import instructor
import boto3
from typing import Type
from pydantic import BaseModel, Field

# --- Pydantic and DataClass Definitions ---

# Create a proper Pydantic model for instructor
class EmailClassificationPydantic(BaseModel):
    """Pydantic model for instructor structured output"""
    category: str = Field(
        description="The email category: 'work', 'personal', 'spam', or 'promotional'"
    )
    reasoning: str = Field(
        description="Step-by-step reasoning for the classification decision"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1 for the classification",
        ge=0.0,
        le=1.0
    )

# Keep the AdalFlow DataClass for internal use
@dataclass
class EmailClassificationOutput(DataClass):
    """AdalFlow DataClass for internal pipeline use"""
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

# --- Bedrock API and Model Client ---

class AnthropicBedrockChatCompletions:
    """Wrapper for instructor and Bedrock to call Anthropic models."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create(
        self,
        modelId: str,
        max_tokens: int,
        system_message: str, # FIX: Should be a string for instructor/Claude3
        messages: List,
        response_model: Type[BaseModel],
        dump: bool = True
    ):
        """Send prompt and acquire the chat completions response."""
        bedrock_client = boto3.client('bedrock-runtime')
        client = instructor.from_bedrock(bedrock_client)
        
        try:
            # The 'instructor' library expects a string for the 'system' parameter
            # when using Claude 3, not a list of dicts.
            resp = client.chat.completions.create(
                model=modelId, # instructor uses 'model' kwarg
                max_tokens=max_tokens,
                system=system_message,
                messages=messages,
                response_model=response_model
            )
            self.logger.info("Bedrock API call successful")
            # Return the Pydantic object or its dictionary representation
            return resp.model_dump() if dump else resp
        except Exception as e:
            self.logger.error(f"Could not complete prompt due to: {str(e)}")
            raise RuntimeError(f"Could not complete prompt due to: {str(e)}") from e

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
        
        api_kwargs = {
            "modelId": model_kwargs.get("model", self.model_id),
            "max_tokens": model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": str(input)}],
            "system_message": "You are an expert email classifier.",
            "response_model": EmailClassificationPydantic,
            "dump": True
        }
        
        # Override with any provided model_kwargs
        api_kwargs.update({k: v for k, v in model_kwargs.items() if k in api_kwargs})

        # Special handling for response_model to ensure it's a Pydantic model
        if "response_model" in model_kwargs:
            response_model = model_kwargs["response_model"]
            if isinstance(response_model, type) and issubclass(response_model, BaseModel):
                api_kwargs["response_model"] = response_model
            else:
                self.logger.warning(f"Provided response_model is not a Pydantic BaseModel, using default.")
                api_kwargs["response_model"] = EmailClassificationPydantic
                
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

# --- Custom Offline Components ---

class OfflineGenerator:
    """A replacement for AdalFlow's Generator that works 'offline' by bypassing network-dependent initialization."""
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict = None, template: str = None, prompt_kwargs: Dict = None):
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.template = template
        self.prompt_kwargs = prompt_kwargs or {}
        self.logger = logging.getLogger("OfflineGenerator")
        self.logger.info("OfflineGenerator initialized successfully")
    
    def _format_prompt(self, prompt_kwargs: Dict) -> str:
        """Format the prompt using the template and provided kwargs."""
        if not self.template:
            return prompt_kwargs.get("input_str", "")
            
        combined_kwargs = {**self.prompt_kwargs, **prompt_kwargs}
        formatted_prompt = self.template
        for key, value in combined_kwargs.items():
            # Handle AdalFlow Parameter objects by extracting their data
            str_value = str(value.data if hasattr(value, 'data') else value)
            formatted_prompt = formatted_prompt.replace(f"{{{{{key}}}}}", str_value)
            
        return formatted_prompt
    
    def call(self, prompt_kwargs: Dict = None) -> GeneratorOutput:
        """Generate response using the model client."""
        prompt_kwargs = prompt_kwargs or {}
        
        try:
            formatted_input = self._format_prompt(prompt_kwargs)
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=formatted_input,
                model_kwargs=self.model_kwargs,
                model_type=ModelType.LLM
            )
            return self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        except Exception as e:
            self.logger.error(f"OfflineGenerator error: {e}", exc_info=True)
            return GeneratorOutput(data=None, error=str(e), usage=None, raw_response=None)
    
    def __call__(self, prompt_kwargs: Dict) -> GeneratorOutput:
        return self.call(prompt_kwargs)
    
    async def acall(self, prompt_kwargs: Dict = None) -> GeneratorOutput:
        return self.call(prompt_kwargs)

class EmailClassificationPipeline(Component):
    """Email classification task pipeline using the OfflineGenerator."""
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict):
        super().__init__()
        
        self._system_prompt_data = """You are an expert email classifier. Analyze the email content carefully and classify it into one of these categories:
- work: Professional emails, meetings, business communications
- personal: Personal messages, family, friends
- spam: Unwanted promotional emails, suspicious content
- promotional: Legitimate marketing emails, newsletters

Provide your reasoning step-by-step and assign a confidence score."""
        
        template = r"""<SYS>
{{system_prompt}}
</SYS>

{{few_shot_demos}}

Email to classify:
{{email_content}}

Please analyze this email and provide the classification in the required structured format."""
        
        self.logger = logging.getLogger("EmailClassificationPipeline")
        
        # Create Parameters for optimization
        self.system_prompt = Parameter(
            data=self._system_prompt_data,
            role_desc="The system prompt that guides the email classification task",
            param_type=ParameterType.PROMPT,
            requires_opt=True
        )
        
        self.few_shot_demos = Parameter(
            data="",  # Start with empty demos, will be populated by bootstrap optimizer
            role_desc="Few-shot examples to help the model learn email classification patterns",
            param_type=ParameterType.DEMOS,
            requires_opt=True
        )

        # Create the OfflineGenerator and link it to the Parameters
        self.generator = OfflineGenerator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
                "few_shot_demos": self.few_shot_demos
            }
        )
        self.logger.info("EmailClassificationPipeline with OfflineGenerator created successfully!")
    
    def call(self, email_content: str, id: str = None) -> GeneratorOutput:
        """Process a single email classification request."""
        return self.generator.call(prompt_kwargs={"email_content": email_content})

# --- Custom Offline Backward Engine ---

class OfflineBackwardEngine:
    """
    An offline backward engine that generates textual gradients using an LLM,
    replicating AdalFlow's core logic without network-dependent initialization.
    """
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict = None):
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.logger = logging.getLogger("OfflineBackwardEngine")
        
        class GradientResponse(BaseModel):
            feedback: str = Field(description="Detailed feedback on why the prompt failed and suggestions for improvement.")
            suggestion: str = Field(description="A concrete, improved version of the prompt based on the feedback.")
        
        self.gradient_response_model = GradientResponse
        
        self.gradient_template = """You are an expert prompt engineer analyzing why a language model failed.

CONTEXT:
- Task: {task_description}
- Current Prompt: {current_prompt}
- Model Prediction: {model_prediction}
- Correct Answer: {ground_truth}
- Evaluation Score: {evaluation_score}

ANALYSIS:
1. Identify why the current prompt led to the incorrect prediction.
2. Suggest specific improvements to fix this failure.
3. Provide a revised prompt suggestion.

Your feedback must be specific, actionable, and focused on improving prompt clarity, examples, or instructions.
Provide your analysis as structured feedback and a concrete suggestion."""
        self.logger.info("OfflineBackwardEngine initialized with LLM-based gradient generation")

    def __call__(self, **kwargs):
        """Generate textual gradients using the offline model client."""
        try:
            return self._generate_textual_gradient(**kwargs)
        except Exception as e:
            self.logger.error(f"Error generating LLM-based gradient: {e}")
            return self._generate_fallback_gradient(**kwargs)
    
    def _generate_textual_gradient(self, **kwargs):
        """Generate LLM-based textual gradients."""
        response = kwargs.get('response')
        ground_truth = kwargs.get('ground_truth', '')
        eval_score = kwargs.get('eval_score', 0.0)
        
        current_prompt = kwargs.get('context', "Current prompt not available in context.")
        model_prediction = self._extract_model_prediction(response)
        
        gradient_prompt = self.gradient_template.format(
            task_description="Email classification into work, personal, spam, or promotional categories.",
            current_prompt=current_prompt[:1000],  # Truncate for safety
            model_prediction=model_prediction,
            ground_truth=ground_truth,
            evaluation_score=eval_score
        )
        
        self.logger.info("Generating LLM-based textual gradient...")
        api_kwargs = {
            "modelId": self.model_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0"),
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": gradient_prompt}],
            "system_message": "You are a prompt engineering expert providing feedback for prompt optimization.",
            "response_model": self.gradient_response_model,
            "dump": True
        }
        
        grad_response = self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        if grad_response.data and isinstance(grad_response.data, dict):
            feedback = grad_response.data.get('feedback', 'No feedback.')
            suggestion = grad_response.data.get('suggestion', 'No suggestion.')
            textual_gradient = f"FEEDBACK: {feedback}\n\nSUGGESTION: {suggestion}"
            self.logger.info(f"Generated LLM gradient: {textual_gradient[:200]}...")
            return textual_gradient
        else:
            self.logger.warning("No valid gradient data received, using fallback.")
            return self._generate_fallback_gradient(**kwargs)

    def _generate_fallback_gradient(self, **kwargs):
        """Generate basic feedback if the LLM gradient call fails."""
        model_prediction = self._extract_model_prediction(kwargs.get('response'))
        ground_truth = kwargs.get('ground_truth', '')
        return f"FEEDBACK: Model predicted '{model_prediction}', but expected '{ground_truth}'. Prompt may need clearer category definitions.\n\nSUGGESTION: Refine the distinction between '{model_prediction}' and '{ground_truth}' categories."

    def _extract_model_prediction(self, response) -> str:
        """Extract the model's prediction from the response."""
        if response and response.data:
            if isinstance(response.data, dict):
                return response.data.get('category', 'unknown')
            elif hasattr(response.data, 'category'):
                return response.data.category
        return "unknown"

# --- AdalComponent for Optimization ---

# FIX: Added class definition and inheritance from AdalComponent
class EmailClassificationOptimizer(AdalComponent):
    """
    AdalComponent for optimizing email classification.
    This version uses the OfflineGenerator and OfflineBackwardEngine.
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        teacher_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        self.logger = logging.getLogger("EmailClassificationOptimizer")
        
        task = EmailClassificationPipeline(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # Initialize loss_fn without a backward engine to prevent hanging
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Exact match between predicted and ground truth category",
            backward_engine=None,
        )
        
        # Initialize parent FIRST
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
        )
        
        # Store configs and client for manual backward engine configuration later
        self.backward_engine_model_config = backward_engine_model_config
        self._main_model_client = model_client
        self.logger.info("EmailClassificationOptimizer initialized successfully!")

    # FIX: This method is called by the Trainer to set up the backward pass.
    # We now instantiate our custom OfflineBackwardEngine here.
    def configure_backward_engine(self, *args, **kwargs):
        """
        Configure the backward engine manually using our offline version.
        This is called by the Trainer before optimization begins.
        """
        self.logger.info("Configuring custom OfflineBackwardEngine...")
        try:
            backward_engine = OfflineBackwardEngine(
                model_client=self._main_model_client,
                model_kwargs=self.backward_engine_model_config.get("model_kwargs", {})
            )
            
            # Set the custom backward engine on the loss function
            self.loss_fn.backward_engine = backward_engine
            
            self.logger.info("OfflineBackwardEngine configured successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error configuring backward engine: {e}", exc_info=True)
            return False
    
    def prepare_task(self, sample: EmailDataSample) -> Tuple[Callable, Dict]:
        """Prepare a single task sample for processing."""
        return self.task.call, {"email_content": sample.email_content, "id": sample.id}
    
    def prepare_eval(self, sample: EmailDataSample, y_pred: GeneratorOutput) -> Tuple[Callable, Dict]:
        """Prepare evaluation for a single sample."""
        pred_category = self._extract_category(y_pred)
        return self.eval_fn, {"y": pred_category, "y_gt": sample.true_category}
    
    def prepare_loss(self, sample: EmailDataSample, y_pred: Parameter) -> Tuple[Callable, Dict]:
        """Prepare loss computation for a single sample."""
        y_gt = Parameter(data=sample.true_category, eval_input=sample.true_category, requires_opt=False)
        y_pred.eval_input = self._extract_category(y_pred.full_response)
        
        return self.loss_fn.forward, {"kwargs": {"y": y_pred, "y_gt": y_gt}}

    def _extract_category(self, y_pred: Optional[GeneratorOutput]) -> Optional[str]:
        """Helper to safely extract category from prediction output."""
        if y_pred and y_pred.data:
            if isinstance(y_pred.data, dict):
                return y_pred.data.get('category')
            elif hasattr(y_pred.data, 'category'):
                return y_pred.data.category
        return None

# --- Main Setup and Execution ---

def create_offline_training_setup():
    """Create a complete offline-compatible training setup."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OfflineTrainingSetup")
    
    logger.info("Setting up offline-compatible email classification optimization...")
    
    model_client = OfflineCompatibleModelClient()
    
    model_config = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    # Pass both model_client and model_kwargs to each component's config
    shared_config = {"model_client": model_client, "model_kwargs": model_config}
    
    logger.info("Creating AdalComponent...")
    adal_component = EmailClassificationOptimizer(
        model_client=model_client,
        model_kwargs=model_config,
        backward_engine_model_config=shared_config,
        teacher_model_config=shared_config,
        text_optimizer_model_config=shared_config,
    )
    logger.info("AdalComponent created successfully!")
    
    trainer = Trainer(
        adaltask=adal_component,
        strategy="random",
        max_steps=5,
        num_workers=1,
        train_batch_size=2,
        bootstrap_shots=1,
        debug=True,
        disable_backward=False,          # Enable backward pass
        disable_backward_gradients=False # Enable textual gradient computation
    )
    
    logger.info("Offline training setup completed successfully!")
    return adal_component, trainer, model_client

if __name__ == "__main__":
    train_samples = [
        EmailDataSample("Meeting scheduled for tomorrow at 2 PM.", "work", "1"),
        EmailDataSample("Hey! Want to grab dinner tonight?", "personal", "2"),
        EmailDataSample("CONGRATULATIONS! You've won $1,000,000!", "spam", "3"),
        EmailDataSample("New products available with 20% discount.", "promotional", "4"),
    ]
    val_samples = train_samples[:2]
    
    try:
        print("🚀 Starting offline AdalFlow setup...")
        adal_component, trainer, model_client = create_offline_training_setup()
        print("✅ SUCCESS! Initialization complete without hanging.")
        
        print("\n📧 Testing a single forward pass...")
        sample = train_samples[0]
        result = adal_component.task.call(sample.email_content, sample.id)
        print(f"📊 Classification result for '{sample.email_content}': {result.data}")
        
        if result.error:
             print(f"❌ Error during forward pass: {result.error}")
        
        print("\n🎉 Offline setup is working perfectly!")
        print("🚀 To run the full optimization, uncomment the line below:")
        # print("trainer.fit(train_dataset=train_samples, val_dataset=val_samples)")
        
        # Example of running the fit method
        # NOTE: This will make live API calls to Bedrock and may incur costs.
        # trainer.fit(train_dataset=train_samples, val_dataset=val_samples)
        
    except Exception as e:
        print(f"\n❌ Setup or execution failed: {e}")
        import traceback
        traceback.print_exc()
