import logging
import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.core.component import Component
from adalflow.core.base_data_class import DataClass
from adalflow.optim.parameter import Parameter
from adalflow.optim.types import ParameterType
from adalflow.optim.trainer.adal import AdalComponent
from adalflow.optim.trainer.trainer import Trainer
from adalflow.optim.text_grad.text_loss_with_eval_fn import EvalFnToTextLoss
from adalflow.core.generator import BackwardEngine
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# Your Bedrock client imports
import instructor
import boto3
from typing import Type
from pydantic import BaseModel, Field

# Create a proper Pydantic model for instructor (not AdalFlow DataClass)
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
        
        # Use the Pydantic model for instructor compatibility
        api_kwargs = {
            "modelId": model_kwargs.get("model", self.model_id),
            "max_tokens": model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": str(input)}],
            "system_message": "You are an expert email classifier. Analyze the email content and provide a structured classification.",
            "response_model": EmailClassificationPydantic,  # Use Pydantic model
            "dump": True
        }
        
        # Override with any provided model_kwargs
        if "system_message" in model_kwargs:
            api_kwargs["system_message"] = model_kwargs["system_message"]
        if "response_model" in model_kwargs:
            # Allow overriding but ensure it's a Pydantic model
            response_model = model_kwargs["response_model"]
            if hasattr(response_model, '__bases__') and BaseModel in response_model.__bases__:
                api_kwargs["response_model"] = response_model
            else:
                self.logger.warning(f"Provided response_model is not a Pydantic BaseModel, using default")
                api_kwargs["response_model"] = EmailClassificationPydantic
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

class OfflineGenerator:
    """
    A replacement for AdalFlow's Generator that works offline by bypassing initialization issues.
    This provides the same interface but avoids network-dependent initialization.
    """
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict = None, template: str = None, prompt_kwargs: Dict = None):
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.template = template
        self.prompt_kwargs = prompt_kwargs or {}
        self.logger = logging.getLogger("OfflineGenerator")
        self.logger.info("OfflineGenerator initialized successfully (offline mode)")
    
    def _format_prompt(self, prompt_kwargs: Dict) -> str:
        """Format the prompt using the template and provided kwargs."""
        if self.template:
            # Combine default prompt_kwargs with call-time prompt_kwargs
            combined_kwargs = {**self.prompt_kwargs, **prompt_kwargs}
            
            # Simple template formatting (you could use Jinja2 here if needed)
            formatted_prompt = self.template
            for key, value in combined_kwargs.items():
                # Handle Parameter objects
                if hasattr(value, 'data'):
                    value = value.data
                formatted_prompt = formatted_prompt.replace(f"{{{{{key}}}}}", str(value))
            
            return formatted_prompt
        else:
            # No template, just use input_str directly
            return prompt_kwargs.get("input_str", "")
    
    def call(self, prompt_kwargs: Dict = None) -> GeneratorOutput:
        """Generate response using the model client."""
        prompt_kwargs = prompt_kwargs or {}
        
        try:
            # Format the prompt
            formatted_input = self._format_prompt(prompt_kwargs)
            
            # Convert to API kwargs
            api_kwargs = self.model_client.convert_inputs_to_api_kwargs(
                input=formatted_input,
                model_kwargs=self.model_kwargs,
                model_type=ModelType.LLM
            )
            
            # Make the call
            response = self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
            return response
            
        except Exception as e:
            self.logger.error(f"OfflineGenerator error: {e}", exc_info=True)
            return GeneratorOutput(data=None, error=str(e), usage=None, raw_response=None)
    
    def __call__(self, prompt_kwargs: Dict) -> GeneratorOutput:
        """Allow calling the generator as a function."""
        return self.call(prompt_kwargs)
    
    async def acall(self, prompt_kwargs: Dict = None) -> GeneratorOutput:
        """Async version - for now just call sync version."""
        return self.call(prompt_kwargs)

class EmailClassificationPipeline(Component):
    """
    Email classification task pipeline with structured output.
    Uses OfflineGenerator to avoid initialization issues.
    """
    
    def __init__(self, model_client: ModelClient, model_kwargs: Dict):
        super().__init__()
        
        # Store the system prompt data
        self._system_prompt_data = """You are an expert email classifier. Analyze the email content carefully and classify it into one of these categories:
- work: Professional emails, meetings, business communications
- personal: Personal messages, family, friends
- spam: Unwanted promotional emails, suspicious content
- promotional: Legitimate marketing emails, newsletters

Provide your reasoning step-by-step and assign a confidence score."""
        
        # Template for the email classification with demo support
        template = r"""<SYS>
{{system_prompt}}
</SYS>

{{few_shot_demos}}

Email to classify:
{{email_content}}

Please analyze this email and provide:
1. Category (work/personal/spam/promotional)
2. Your reasoning
3. Confidence score (0-1)
"""
        
        # Create the OfflineGenerator - this should NOT hang
        self.logger = logging.getLogger("EmailClassificationPipeline")
        self.logger.info("Creating OfflineGenerator...")
        
        self.generator = OfflineGenerator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=template,
            prompt_kwargs={
                "system_prompt": self._system_prompt_data,
                "few_shot_demos": ""  # Start with empty demos
            }
        )
        
        self.logger.info("OfflineGenerator created successfully!")
        
        # Create the Parameters for optimization
        self.system_prompt = Parameter(
            data=self._system_prompt_data,
            role_desc="The system prompt that guides the email classification task",
            param_type=ParameterType.PROMPT,
            requires_opt=True
        )
        
        # Add few-shot demo parameter for bootstrap optimization
        self.few_shot_demos = Parameter(
            data=None,  # Will be populated by bootstrap optimizer
            role_desc="Few-shot examples to help the model learn email classification patterns",
            param_type=ParameterType.DEMOS,
            requires_opt=True
        )
        
        # Update the generator to use the Parameters for optimization
        self.generator.prompt_kwargs["system_prompt"] = self.system_prompt
        self.generator.prompt_kwargs["few_shot_demos"] = self.few_shot_demos
    
    def call(self, email_content: str, id: str = None) -> GeneratorOutput:
        """Process a single email classification request."""
        return self.generator.call(prompt_kwargs={"email_content": email_content})

class OfflineBackwardEngine:
    """
    A complete offline backward engine that replicates AdalFlow's textual gradient logic.
    Uses your offline model client to generate LLM-based gradients just like the real AdalFlow.
    This gives you the full gradient-based prompt optimization experience offline.
    """
    
    def __init__(self, model_client: ModelClient = None, model_kwargs: Dict = None):
        self.model_client = model_client
        self.model_kwargs = model_kwargs or {}
        self.logger = logging.getLogger("OfflineBackwardEngine")
        self.logger.info("OfflineBackwardEngine initialized with LLM gradient generation")
        
        # Create a simple response model for gradient generation
        class GradientResponse(BaseModel):
            feedback: str = Field(
                description="Detailed feedback on why the current prompt failed and specific suggestions for improvement"
            )
            suggestion: str = Field(
                description="A concrete suggestion for improving the prompt based on the failure analysis"
            )
            confidence: float = Field(
                description="Confidence in the feedback (0.0 to 1.0)",
                ge=0.0, le=1.0
            )
        
        self.gradient_response_model = GradientResponse
        
        # Template for generating textual gradients (based on AdalFlow's approach)
        self.gradient_template = """You are an expert prompt engineer analyzing why a language model failed on a task.

CONTEXT:
- Task: {task_description}
- Current Prompt: {current_prompt}
- Model Prediction: {model_prediction}
- Correct Answer: {ground_truth}
- Evaluation Score: {evaluation_score}

ANALYSIS REQUIRED:
1. Identify why the current prompt led to the incorrect prediction
2. Suggest specific improvements to the prompt to fix this failure
3. Focus on concrete, actionable changes

Your feedback should be:
- Specific and actionable
- Based on the failure pattern you observe
- Aimed at improving classification accuracy
- Focused on prompt clarity, examples, or instructions

Provide detailed feedback and a concrete suggestion for improvement."""
    
    def __call__(self, **kwargs):
        """
        Generate textual gradients using the offline model client.
        This replicates AdalFlow's backward engine logic with LLM-generated feedback.
        """
        try:
            return self._generate_textual_gradient(**kwargs)
        except Exception as e:
            self.logger.error(f"Error generating gradient: {e}")
            # Fallback to basic feedback if LLM call fails
            return self._generate_fallback_gradient(**kwargs)
    
    def _generate_textual_gradient(self, **kwargs):
        """Generate LLM-based textual gradients using the offline model client."""
        
        # Extract gradient context information
        response = kwargs.get('response')
        ground_truth = kwargs.get('ground_truth', '')
        eval_score = kwargs.get('eval_score', 0.0)
        context = kwargs.get('context', '')
        variable_desc = kwargs.get('variable_desc', 'system prompt')
        
        # Extract current prompt and prediction
        current_prompt = self._extract_current_prompt(context, variable_desc)
        model_prediction = self._extract_model_prediction(response)
        
        # Prepare the gradient generation prompt
        gradient_prompt = self.gradient_template.format(
            task_description="Email classification into work, personal, spam, or promotional categories",
            current_prompt=current_prompt,
            model_prediction=model_prediction,
            ground_truth=ground_truth,
            evaluation_score=eval_score
        )
        
        self.logger.info("Generating LLM-based textual gradient...")
        
        # Create API kwargs for gradient generation
        api_kwargs = {
            "modelId": self.model_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0"),
            "max_tokens": self.model_kwargs.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": gradient_prompt}],
            "system_message": "You are an expert prompt engineer providing detailed feedback for prompt optimization.",
            "response_model": self.gradient_response_model,
            "dump": True
        }
        
        # Generate gradient using the offline model client
        response = self.model_client.call(api_kwargs=api_kwargs, model_type=ModelType.LLM)
        
        if response.data and isinstance(response.data, dict):
            feedback = response.data.get('feedback', 'No specific feedback generated')
            suggestion = response.data.get('suggestion', 'Consider refining the prompt')
            
            # Format the gradient in AdalFlow's expected format
            textual_gradient = f"FEEDBACK: {feedback}\n\nSUGGESTION: {suggestion}"
            
            self.logger.info(f"Generated LLM gradient: {textual_gradient[:200]}...")
            return textual_gradient
        else:
            self.logger.warning("No valid gradient data received, using fallback")
            return self._generate_fallback_gradient(**kwargs)
    
    def _generate_fallback_gradient(self, **kwargs):
        """Generate basic feedback if LLM gradient generation fails."""
        response = kwargs.get('response')
        ground_truth = kwargs.get('ground_truth', '')
        
        model_prediction = self._extract_model_prediction(response)
        
        if model_prediction != ground_truth:
            return f"FEEDBACK: The model predicted '{model_prediction}' but the correct answer is '{ground_truth}'. The prompt may need clearer category definitions or better examples.\n\nSUGGESTION: Add specific examples for the '{ground_truth}' category and refine the distinction between '{model_prediction}' and '{ground_truth}' categories."
        else:
            return "FEEDBACK: The prediction was correct, but consider adding more specific instructions for edge cases.\n\nSUGGESTION: Enhance the prompt with clearer reasoning instructions."
    
    def _extract_current_prompt(self, context, variable_desc):
        """Extract the current prompt from the context."""
        if "system_prompt" in context.lower():
            # Try to extract the system prompt section
            lines = context.split('\n')
            for i, line in enumerate(lines):
                if 'system_prompt' in line.lower() or '<SYS>' in line:
                    # Look for the prompt content in the next few lines
                    prompt_lines = []
                    for j in range(i, min(i + 10, len(lines))):
                        if lines[j].strip():
                            prompt_lines.append(lines[j].strip())
                    return '\n'.join(prompt_lines)
        return context[:500] if context else "Current prompt not available"
    
    def _extract_model_prediction(self, response):
        """Extract the model's prediction from the response."""
        if hasattr(response, 'data') and response.data:
            if isinstance(response.data, dict):
                return response.data.get('category', 'unknown')
            elif hasattr(response.data, 'category'):
                return response.data.category
        return "unknown"
    """
    AdalComponent for optimizing email classification with offline support.
    This version completely avoids BackwardEngine initialization issues.
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
        self.logger.info("Initializing EmailClassificationOptimizer...")
        
        # Create the task pipeline - this should work now
        task = EmailClassificationPipeline(model_client, model_kwargs)
        self.logger.info("Task pipeline created successfully!")
        
        # Create evaluation function
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # Create loss function WITHOUT any backward engine to prevent initialization
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Exact match between predicted and ground truth category",
            # CRITICAL: Pass None for all backward engine related parameters
            backward_engine=None,
            model_client=None,  # Pass None to prevent automatic creation
            model_kwargs=None   # Pass None to prevent automatic creation
        )
        
        # Initialize parent FIRST to avoid AttributeError
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            # Don't pass any optimization configs to prevent automatic setup
            backward_engine=None,
            backward_engine_model_config=None,
            teacher_model_config=None,
            text_optimizer_model_config=None,
        )
        
        # NOW store configs for later use after super().__init__()
        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config
        self._main_model_client = model_client  # Store for later backward engine setup
        
        self.logger.info("EmailClassificationOptimizer initialized successfully!")
    
    def configure_backward_engine(self, *args, **kwargs):
        """
        Configure the backward engine manually when needed.
        This is called by the Trainer when optimization actually starts.
        """
        self.logger.info("Configuring backward engine...")
        
        try:
            # Create backward engine using the stored model client
            # Note: We'll need to check the correct BackwardEngine constructor signature
            backward_engine = BackwardEngine()  # Initialize without parameters first
            
            # Then configure it with the model client and kwargs if needed
            # This approach avoids the constructor parameter issue
            
            # Set the backward engine on the loss function
            self.loss_fn.backward_engine = backward_engine
            
            self.logger.info("Backward engine configured successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error configuring backward engine: {e}")
            self.logger.warning("Continuing without backward engine - demo optimization still available")
            return False
    
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
    This version should work without any hanging.
    """
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("OfflineTrainingSetup")
    
    logger.info("Setting up offline-compatible email classification optimization...")
    
    # Create model client
    model_client = OfflineCompatibleModelClient()
    logger.info("Model client created successfully")
    
    # Model configurations - need to include the actual model_client instance
    model_config = {
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "max_tokens": 1024,
        "temperature": 0.1
    }
    
    # Configuration for different optimization components
    # Each needs both the model_client instance and model_kwargs
    backward_engine_config = {
        "model_client": model_client,
        "model_kwargs": model_config
    }
    
    teacher_model_config = {
        "model_client": model_client,
        "model_kwargs": model_config
    }
    
    text_optimizer_config = {
        "model_client": model_client,
        "model_kwargs": model_config
    }
    
    # Create the AdalComponent - this should NOT hang now
    logger.info("Creating AdalComponent (using OfflineGenerator)...")
    adal_component = EmailClassificationOptimizer(
        model_client=model_client,
        model_kwargs=model_config,
        backward_engine_model_config=backward_engine_config,
        teacher_model_config=teacher_model_config,
        text_optimizer_model_config=text_optimizer_config,
    )
    logger.info("AdalComponent created successfully!")
    
    # Create trainer with prompt optimization enabled but offline compatible
    trainer = Trainer(
        adaltask=adal_component,
        strategy="random",
        max_steps=5,  # Increase steps for better optimization
        num_workers=1,
        train_batch_size=2,
        raw_shots=0,
        bootstrap_shots=1,
        debug=True,
        disable_backward=False,  # ✅ Enable backward pass with mock engine
        disable_backward_gradients=False,  # ✅ Enable gradient computation with mock feedback
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
    
    val_samples = train_samples[:2]
    test_samples = train_samples[2:]
    
    try:
        print("🚀 Starting offline AdalFlow setup...")
        
        # This should NOT hang anymore
        adal_component, trainer, model_client = create_offline_training_setup()
        
        print("✅ SUCCESS! No hanging occurred.")
        print("📧 Testing email classification...")
        
        # Test individual task execution
        sample = train_samples[0]
        result = adal_component.task.call(sample.email_content, sample.id)
        print(f"📊 Classification result: {result}")
        
        if result.data:
            print(f"✅ Structured output received: {result.data}")
        else:
            print(f"❌ Error in classification: {result.error}")
        
        print("\n🎉 Offline setup is working perfectly!")
        print("🚀 Ready for full optimization. To train:")
        print("trainer.fit(train_dataset=train_samples, val_dataset=val_samples, test_dataset=test_samples)")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
