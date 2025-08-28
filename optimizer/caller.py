"""
AdalFlow + AWS Bedrock Text Classification with Prompt Optimization

This module provides a complete setup for text classification using AdalFlow's prompt
optimization capabilities with AWS Bedrock models and instructor for structured output.

Features:
- Custom Bedrock ModelClient for AdalFlow integration
- Text classification with structured output using instructor
- Prompt optimization using AdalFlow's Parameter and TGDOptimizer
- Clean separation of concerns with proper documentation
"""

import logging
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple, Callable
from pydantic import BaseModel

# Core AdalFlow imports
import adalflow as adal
from adalflow.core import Component, Generator, DataClass
from adalflow.optim import (
    Parameter, 
    ParameterType, 
    AdalComponent, 
    Trainer,
    TGDOptimizer,
    DemoOptimizer
)
from adalflow.optim.types import EvalFnToTextLoss
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# Your existing Bedrock caller
from bedrock_caller import AnthropicBedrockChatCompletions


# ===============================================================================
# BEDROCK MODEL CLIENT FOR ADALFLOW
# ===============================================================================

class BedrockModelClient(adal.ModelClient):
    """
    Custom ModelClient that integrates your AnthropicBedrockChatCompletions
    with AdalFlow's optimization framework.
    
    This client maintains compatibility with AdalFlow's Parameter optimization
    while using your existing Bedrock integration.
    """
    
    def __init__(self):
        """Initialize the Bedrock client wrapper."""
        super().__init__()
        self.bedrock_client = AnthropicBedrockChatCompletions()
        self.logger = logging.getLogger(__name__)
    
    def call(
        self,
        api_kwargs: Dict[str, Any],
        model_type: adal.ModelType = adal.ModelType.LLM
    ) -> Any:
        """
        Call method for inference mode.
        
        Args:
            api_kwargs: Contains model parameters including messages, system prompt, etc.
            model_type: Type of model being called (LLM in this case)
            
        Returns:
            Response from the model
        """
        try:
            # Extract parameters from api_kwargs
            messages = api_kwargs.get("messages", [])
            system_message = api_kwargs.get("system", "")
            max_tokens = api_kwargs.get("max_tokens", 4000)
            model_id = api_kwargs.get("model", "anthropic.claude-3-sonnet-20240229-v1:0")
            response_model = api_kwargs.get("response_model")
            
            # Call your existing Bedrock client
            response = self.bedrock_client.create(
                modelId=model_id,
                max_tokens=max_tokens,
                system_message=system_message,
                messages=messages,
                response_model=response_model,
                dump=False  # Return pydantic object for AdalFlow compatibility
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Bedrock call failed: {str(e)}")
            raise
    
    async def acall(
        self,
        api_kwargs: Dict[str, Any],
        model_type: adal.ModelType = adal.ModelType.LLM
    ) -> Any:
        """Async version of call method."""
        # For now, just call the sync version
        # You can implement true async later if needed
        return self.call(api_kwargs, model_type)


# ===============================================================================
# DATA MODELS
# ===============================================================================

@dataclass
class ClassificationData(DataClass):
    """
    DataClass for text classification input/output.
    
    This follows AdalFlow's DataClass pattern for structured LLM interactions.
    """
    text: str = field(
        metadata={"desc": "The input text to be classified"},
        default=None
    )
    reasoning: str = field(
        metadata={"desc": "Step-by-step reasoning for the classification"},
        default=None
    )
    predicted_class: str = field(
        metadata={"desc": "The predicted class label"},
        default=None
    )
    confidence: float = field(
        metadata={"desc": "Confidence score between 0 and 1"},
        default=None
    )
    
    # Define input and output fields for AdalFlow optimization
    __input_fields__ = ["text"]
    __output_fields__ = ["reasoning", "predicted_class", "confidence"]


class ClassificationOutput(BaseModel):
    """Pydantic model for structured output with instructor."""
    reasoning: str
    predicted_class: str
    confidence: float


# ===============================================================================
# CLASSIFICATION COMPONENT
# ===============================================================================

class TextClassifier(Component):
    """
    AdalFlow Component for text classification with prompt optimization.
    
    This component uses your Bedrock client and supports both training and inference modes.
    The prompts can be optimized using AdalFlow's Parameter system.
    """
    
    def __init__(
        self, 
        model_client: BedrockModelClient,
        model_kwargs: Dict[str, Any],
        class_labels: List[str],
        class_descriptions: Optional[List[str]] = None
    ):
        """
        Initialize the text classifier.
        
        Args:
            model_client: The Bedrock model client
            model_kwargs: Model parameters (model_id, max_tokens, etc.)
            class_labels: List of possible class labels
            class_descriptions: Optional descriptions for each class
        """
        super().__init__()
        
        self.class_labels = class_labels
        self.class_descriptions = class_descriptions or class_labels
        
        # Create optimizable parameters
        self.system_prompt = Parameter(
            data=self._create_default_system_prompt(),
            role_desc="System prompt that instructs the model on classification task",
            requires_opt=True,
            param_type=ParameterType.PROMPT
        )
        
        self.few_shot_demos = Parameter(
            data=None,
            role_desc="Few-shot examples for in-context learning",
            requires_opt=True,
            param_type=ParameterType.DEMOS
        )
        
        # Template for the complete prompt
        self.prompt_template = """<START_OF_SYSTEM_MESSAGE>
{{system_prompt}}

Available Classes:
{% for i in range(class_labels|length) %}
{{ i }}. {{ class_labels[i] }}: {{ class_descriptions[i] }}
{% endfor %}

Your response must include:
1. Reasoning: Step-by-step analysis
2. Predicted_class: One of the available classes
3. Confidence: Float between 0 and 1

{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_MESSAGE>

<START_OF_USER>
Text to classify: {{input_text}}
<END_OF_USER>"""
        
        # Create the generator with your Bedrock client
        self.generator = Generator(
            model_client=model_client,
            model_kwargs={
                **model_kwargs,
                "response_model": ClassificationOutput
            },
            template=self.prompt_template,
            prompt_kwargs={
                "system_prompt": self.system_prompt,
                "few_shot_demos": self.few_shot_demos,
                "class_labels": self.class_labels,
                "class_descriptions": self.class_descriptions,
            },
            use_cache=True
        )
    
    def _create_default_system_prompt(self) -> str:
        """Create a default system prompt for text classification."""
        return (
            "You are an expert text classifier. Analyze the given text and classify it "
            "into one of the provided categories. Always provide your reasoning step-by-step "
            "before making the final classification. Be confident but honest about uncertainty."
        )
    
    def call(self, input_text: str, id: Optional[str] = None) -> ClassificationData:
        """
        Inference mode - classify text without optimization tracking.
        
        Args:
            input_text: Text to classify
            id: Optional ID for the request
            
        Returns:
            ClassificationData with predictions
        """
        try:
            # Generate response using the generator
            result = self.generator.call(input_text=input_text, id=id)
            
            # Handle the response based on success/failure
            if result.error:
                self.logger.error(f"Classification failed: {result.error}")
                return ClassificationData(
                    text=input_text,
                    reasoning="Error in classification",
                    predicted_class="unknown",
                    confidence=0.0
                )
            
            # Extract structured output
            output = result.data
            return ClassificationData(
                text=input_text,
                reasoning=output.reasoning,
                predicted_class=output.predicted_class,
                confidence=output.confidence
            )
            
        except Exception as e:
            self.logger.error(f"Classification error: {str(e)}")
            return ClassificationData(
                text=input_text,
                reasoning=f"Error: {str(e)}",
                predicted_class="error",
                confidence=0.0
            )
    
    def forward(self, input_text: str, id: Optional[str] = None) -> Parameter:
        """
        Training mode - classify text with Parameter tracking for optimization.
        
        Args:
            input_text: Text to classify
            id: Optional ID for few-shot learning tracing
            
        Returns:
            Parameter containing the output for gradient computation
        """
        # In training mode, the generator returns a Parameter
        result = self.generator(input_text=input_text, id=id)
        return result


# ===============================================================================
# EVALUATION METRICS
# ===============================================================================

def classification_accuracy(y_pred: ClassificationData, y_gt: ClassificationData) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_pred: Predicted classification data
        y_gt: Ground truth classification data
        
    Returns:
        Accuracy score (1.0 if correct, 0.0 if incorrect)
    """
    if y_pred.predicted_class == y_gt.predicted_class:
        return 1.0
    return 0.0


def confidence_weighted_accuracy(y_pred: ClassificationData, y_gt: ClassificationData) -> float:
    """
    Compute confidence-weighted accuracy.
    
    Args:
        y_pred: Predicted classification data
        y_gt: Ground truth classification data
        
    Returns:
        Weighted accuracy score
    """
    base_score = classification_accuracy(y_pred, y_gt)
    confidence_weight = y_pred.confidence if y_pred.confidence else 0.5
    return base_score * confidence_weight


# ===============================================================================
# OPTIMIZATION COMPONENT
# ===============================================================================

class TextClassificationOptimizer(AdalComponent):
    """
    AdalComponent for optimizing text classification prompts.
    
    This component defines the complete optimization pipeline including:
    - Task definition (TextClassifier)
    - Evaluation functions
    - Training/validation steps
    - Optimizer configuration
    """
    
    def __init__(
        self,
        model_client: BedrockModelClient,
        model_kwargs: Dict[str, Any],
        class_labels: List[str],
        class_descriptions: Optional[List[str]] = None,
        teacher_model_config: Optional[Dict] = None,
        backward_engine_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None
    ):
        """
        Initialize the classification optimizer.
        
        Args:
            model_client: Bedrock model client
            model_kwargs: Model parameters
            class_labels: Available class labels
            class_descriptions: Optional class descriptions
            teacher_model_config: Config for teacher model (for few-shot optimization)
            backward_engine_model_config: Config for backward engine
            text_optimizer_model_config: Config for text optimizer
        """
        # Initialize the task component
        task = TextClassifier(
            model_client=model_client,
            model_kwargs=model_kwargs,
            class_labels=class_labels,
            class_descriptions=class_descriptions
        )
        
        # Initialize evaluation function
        eval_fn = classification_accuracy
        
        # Initialize loss function for text optimization
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Classification accuracy evaluator"
        )
        
        # Initialize the parent with all components
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            teacher_model_config=teacher_model_config,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config
        )
        
        self.logger = logging.getLogger(__name__)
    
    def handle_one_task_sample(self, sample: ClassificationData) -> Tuple[Component, Dict]:
        """
        Handle a single task sample for training/evaluation.
        
        Args:
            sample: Classification data sample
            
        Returns:
            Tuple of (task_component, task_kwargs)
        """
        return self.task, {
            "input_text": sample.text,
            "id": getattr(sample, 'id', None)
        }
    
    def handle_one_loss_sample(
        self, 
        sample: ClassificationData, 
        y_pred: Parameter
    ) -> Tuple[Callable, Dict]:
        """
        Handle a single loss computation.
        
        Args:
            sample: Ground truth classification data
            y_pred: Predicted output parameter
            
        Returns:
            Tuple of (loss_function, loss_kwargs)
        """
        return self.loss_fn.forward, {
            "kwargs": {
                "y": y_pred,
                "y_gt": Parameter(
                    data=sample,
                    role_desc="Ground truth classification",
                    requires_opt=False
                )
            }
        }
    
    def configure_optimizers(self) -> List[adal.Optimizer]:
        """
        Configure optimizers for prompt optimization.
        
        Returns:
            List of configured optimizers
        """
        optimizers = []
        
        # Configure text optimizer for system prompt optimization
        if self.text_optimizer_model_config:
            text_optimizer = TGDOptimizer(
                model_client=self.text_optimizer_model_config["model_client"],
                model_kwargs=self.text_optimizer_model_config["model_kwargs"]
            )
            optimizers.append(text_optimizer)
        
        # Configure demo optimizer for few-shot learning
        demo_optimizer = DemoOptimizer()
        optimizers.append(demo_optimizer)
        
        return optimizers
    
    def configure_teacher_generator(self) -> Optional[Generator]:
        """Configure teacher generator for few-shot optimization."""
        if self.teacher_model_config:
            return Generator(
                model_client=self.teacher_model_config["model_client"],
                model_kwargs=self.teacher_model_config["model_kwargs"],
                template="You are a teacher providing high-quality examples. {{input_str}}"
            )
        return None


# ===============================================================================
# TRAINING AND OPTIMIZATION UTILITIES
# ===============================================================================

class ClassificationTrainer:
    """
    High-level trainer for text classification optimization.
    
    This class provides a clean interface for setting up and running
    prompt optimization experiments.
    """
    
    def __init__(
        self,
        class_labels: List[str],
        class_descriptions: Optional[List[str]] = None,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens: int = 4000
    ):
        """
        Initialize the classification trainer.
        
        Args:
            class_labels: List of class labels
            class_descriptions: Optional descriptions for each class
            model_id: Bedrock model ID to use
            max_tokens: Maximum tokens for generation
        """
        self.class_labels = class_labels
        self.class_descriptions = class_descriptions
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(__name__)
        
        # Initialize model client
        self.model_client = BedrockModelClient()
        self.model_kwargs = {
            "model": model_id,
            "max_tokens": max_tokens
        }
        
    def create_optimizer(
        self,
        teacher_model_config: Optional[Dict] = None,
        text_optimizer_model_config: Optional[Dict] = None
    ) -> TextClassificationOptimizer:
        """
        Create a classification optimizer with the specified configuration.
        
        Args:
            teacher_model_config: Configuration for teacher model
            text_optimizer_model_config: Configuration for text optimizer
            
        Returns:
            Configured TextClassificationOptimizer
        """
        # Default teacher config if not provided
        if teacher_model_config is None:
            teacher_model_config = {
                "model_client": self.model_client,
                "model_kwargs": {
                    "model": "anthropic.claude-3-opus-20240229-v1:0",  # Use stronger model as teacher
                    "max_tokens": self.max_tokens
                }
            }
        
        # Default text optimizer config if not provided
        if text_optimizer_model_config is None:
            text_optimizer_model_config = {
                "model_client": self.model_client,
                "model_kwargs": self.model_kwargs
            }
        
        return TextClassificationOptimizer(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            class_labels=self.class_labels,
            class_descriptions=self.class_descriptions,
            teacher_model_config=teacher_model_config,
            text_optimizer_model_config=text_optimizer_model_config
        )
    
    def train(
        self,
        train_data: List[ClassificationData],
        val_data: List[ClassificationData],
        max_steps: int = 10,
        optimization_order: str = "sequential"  # "sequential" or "mixed"
    ) -> Dict[str, Any]:
        """
        Train and optimize the classification pipeline.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            max_steps: Maximum optimization steps
            optimization_order: How to order optimization ("sequential" or "mixed")
            
        Returns:
            Dictionary with training results and metrics
        """
        self.logger.info("Starting classification optimization...")
        
        # Create optimizer
        optimizer_component = self.create_optimizer()
        
        # Create trainer
        trainer = Trainer(
            adaltask=optimizer_component,
            strategy="constrained",  # Use constrained strategy for better convergence
            max_steps=max_steps,
            num_workers=1,  # Keep it simple for now
            save_traces=True
        )
        
        try:
            # Run training
            trainer.fit(
                train_loader=train_data,
                val_loader=val_data,
                test_loader=None,
                resume_from_ckpt=None
            )
            
            # Get training results
            results = {
                "final_accuracy": trainer.val_score,
                "training_steps": trainer.step,
                "optimization_history": trainer.training_history,
                "optimized_prompts": {
                    "system_prompt": optimizer_component.task.system_prompt.data,
                    "few_shot_demos": optimizer_component.task.few_shot_demos.data
                }
            }
            
            self.logger.info(f"Optimization completed. Final accuracy: {results['final_accuracy']:.3f}")
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(
        self, 
        test_data: List[ClassificationData],
        optimized_component: Optional[TextClassificationOptimizer] = None
    ) -> Dict[str, float]:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_data: Test dataset
            optimized_component: Optional optimized component to use
            
        Returns:
            Dictionary with evaluation metrics
        """
        if optimized_component is None:
            # Create baseline component
            optimized_component = self.create_optimizer()
        
        correct = 0
        total = len(test_data)
        confidence_scores = []
        
        for sample in test_data:
            try:
                # Get prediction
                prediction = optimized_component.task.call(sample.text)
                
                # Check accuracy
                if prediction.predicted_class == sample.predicted_class:
                    correct += 1
                
                confidence_scores.append(prediction.confidence)
                
            except Exception as e:
                self.logger.error(f"Evaluation error for sample: {str(e)}")
        
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        return {
            "accuracy": accuracy,
            "average_confidence": avg_confidence,
            "total_samples": total,
            "correct_predictions": correct
        }


# ===============================================================================
# USAGE EXAMPLE
# ===============================================================================

def example_usage():
    """
    Example of how to use the classification system.
    """
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Define classification problem
    class_labels = ["positive", "negative", "neutral"]
    class_descriptions = [
        "Positive sentiment or opinion",
        "Negative sentiment or opinion", 
        "Neutral or objective statement"
    ]
    
    # Create sample data
    train_data = [
        ClassificationData(text="I love this product!", predicted_class="positive"),
        ClassificationData(text="This is terrible", predicted_class="negative"),
        ClassificationData(text="The weather is cloudy", predicted_class="neutral"),
        # Add more training samples...
    ]
    
    val_data = [
        ClassificationData(text="Great experience!", predicted_class="positive"),
        ClassificationData(text="Not good at all", predicted_class="negative"),
        # Add more validation samples...
    ]
    
    # Initialize trainer
    trainer = ClassificationTrainer(
        class_labels=class_labels,
        class_descriptions=class_descriptions,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    try:
        # Train and optimize
        results = trainer.train(
            train_data=train_data,
            val_data=val_data,
            max_steps=5
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Final accuracy: {results['final_accuracy']:.3f}")
        logger.info(f"Optimized system prompt: {results['optimized_prompts']['system_prompt']}")
        
        # Test inference with optimized model
        classifier = trainer.create_optimizer()
        test_text = "This movie was amazing!"
        prediction = classifier.task.call(test_text)
        
        logger.info(f"Test prediction for '{test_text}':")
        logger.info(f"  Class: {prediction.predicted_class}")
        logger.info(f"  Confidence: {prediction.confidence:.3f}")
        logger.info(f"  Reasoning: {prediction.reasoning}")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise


if __name__ == "__main__":
    example_usage()