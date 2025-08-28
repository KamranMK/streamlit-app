"""
AdalFlow Prompt Optimization System for Text Classification with AWS Bedrock
=====================================================================

This module provides a comprehensive implementation of AdalFlow's prompt optimization
framework integrated with AWS Bedrock via instructor for structured text classification.

Features:
- Custom ModelClient adapter for AWS Bedrock
- DataClass-based structured outputs
- Component-based task pipeline architecture
- Automatic prompt optimization using Text-Grad and Bootstrap optimizers
- Full training and evaluation pipeline

Requirements:
- adalflow==1.1.2
- boto3
- instructor
- pydantic
"""

import os
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Tuple, Callable, Any, Type, Literal
from pydantic import BaseModel

# Specific AdalFlow imports to avoid circular import issues
from adalflow.core.model_client import ModelClient
from adalflow.core.base_data_class import DataClass
from adalflow.core.component import Component
from adalflow.core.types import Parameter, ParameterType
from adalflow.components.output_parsers.dataclass_parser import DataClassParser
from adalflow.core.generator import Generator
from adalflow.core.prompt_builder import Prompt
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from adalflow.optim.text_grad.llm_text_loss import EvalFnToTextLoss
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer
from adalflow.optim.few_shot.bootstrap_optimizer import BootstrapFewShot
from adalflow.core.trainer import Trainer
from adalflow.core.component import AdalComponent

# Import your existing Bedrock client
from your_bedrock_caller import AnthropicBedrockChatCompletions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockModelClient(ModelClient):
    """
    Custom ModelClient adapter for AWS Bedrock using instructor.
    
    This adapter integrates your existing AnthropicBedrockChatCompletions
    with AdalFlow's ModelClient interface for seamless prompt optimization.
    """
    
    def __init__(self, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"):
        """
        Initialize the Bedrock model client.
        
        Args:
            model_id: AWS Bedrock model identifier
        """
        super().__init__()
        self.model_id = model_id
        self.bedrock_client = AnthropicBedrockChatCompletions()
        self.logger = logging.getLogger(__name__)
    
    def init_sync_client(self):
        """Initialize synchronous client - handled by AnthropicBedrockChatCompletions"""
        pass
    
    def parse_chat_completion(self, completion: Any) -> Dict[str, Any]:
        """
        Parse the completion response from Bedrock.
        
        Args:
            completion: Response from Bedrock API
            
        Returns:
            Parsed response in AdalFlow format
        """
        if isinstance(completion, dict):
            # Already parsed by instructor
            return {
                "message": {"content": str(completion)},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": self.model_id
            }
        else:
            # Pydantic model response
            return {
                "message": {"content": completion.model_dump_json() if hasattr(completion, 'model_dump_json') else str(completion)},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": self.model_id
            }
    
    def call(self, api_kwargs: Dict[str, Any], model_type: str = "chat") -> Any:
        """
        Make API call to AWS Bedrock via instructor.
        
        Args:
            api_kwargs: API parameters including messages, system_message, etc.
            model_type: Type of model call (always "chat" for our use case)
            
        Returns:
            Parsed response from the model
        """
        try:
            # Extract parameters from api_kwargs
            messages = api_kwargs.get("messages", [])
            system_message = api_kwargs.get("system", "")
            max_tokens = api_kwargs.get("max_tokens", 1024)
            
            # For structured output, we need a response model
            # If not provided, use a simple string response
            response_model = api_kwargs.get("response_model", str)
            
            # Make the call using your existing Bedrock client
            response = self.bedrock_client.create(
                modelId=self.model_id,
                max_tokens=max_tokens,
                system_message=system_message,
                messages=messages,
                response_model=response_model,
                dump=False  # Get the Pydantic object directly
            )
            
            return self.parse_chat_completion(response)
            
        except Exception as e:
            self.logger.error(f"Bedrock API call failed: {str(e)}")
            raise RuntimeError(f"Bedrock API call failed: {str(e)}")


@dataclass
class ClassificationData(DataClass):
    """
    Base data class for text classification tasks.
    
    Attributes:
        text: Input text to classify
        label: Ground truth label (for training/evaluation)
    """
    text: str = field(metadata={"desc": "The text to be classified"})
    label: Optional[str] = field(default=None, metadata={"desc": "The true label"})
    
    __input_fields__ = ["text"]
    __output_fields__ = []


@dataclass
class ClassificationOutput(DataClass):
    """
    Structured output format for classification with reasoning.
    
    Attributes:
        rationale: Step-by-step reasoning for the classification
        predicted_label: The predicted class label
        confidence: Confidence score (0.0 to 1.0)
    """
    rationale: str = field(
        metadata={"desc": "Your step-by-step reasoning for the classification"}
    )
    predicted_label: str = field(
        metadata={"desc": "The predicted class label"}
    )
    confidence: float = field(
        default=0.8,
        metadata={"desc": "Confidence score between 0.0 and 1.0"}
    )
    
    __input_fields__ = []
    __output_fields__ = ["rationale", "predicted_label", "confidence"]


class TextClassifier(Component):
    """
    Text Classification Component using AdalFlow architecture.
    
    This component handles the core classification logic with structured outputs,
    optimizable prompts, and few-shot demonstration support.
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict[str, Any],
        class_labels: List[str],
        class_descriptions: Optional[List[str]] = None
    ):
        """
        Initialize the text classifier.
        
        Args:
            model_client: AdalFlow model client (our Bedrock adapter)
            model_kwargs: Model-specific parameters
            class_labels: List of possible class labels
            class_descriptions: Optional descriptions for each class
        """
        super().__init__()
        
        self.class_labels = class_labels
        self.class_descriptions = class_descriptions or class_labels
        
        # Create task description template
        self.task_template = self._create_task_template()
        
        # Set up data classes
        self.input_data_class = ClassificationData
        self.output_data_class = ClassificationOutput
        
        # Initialize parser for structured outputs
        self.parser = DataClassParser(
            data_class=self.output_data_class,
            return_data_class=True,
            format_type="yaml"
        )
        
        # Create the main prompt template
        self.template = self._create_prompt_template()
        
        # Set up optimizable parameters
        self.prompt_kwargs = self._setup_parameters()
        
        # Initialize the generator
        self.llm = Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            prompt_kwargs=self.prompt_kwargs,
            template=self.template,
            output_processors=self.parser,
            use_cache=True
        )
    
    def _create_task_template(self) -> str:
        """Create the task description template with class information."""
        return """You are an expert text classifier. Your task is to classify the given text into one of the following categories:

Classes:
{% for label, desc in class_info %}
- {{ label }}: {{ desc }}
{% endfor %}

Instructions:
1. Read the text carefully
2. Think step-by-step about which category best fits
3. Provide your reasoning
4. Make your final classification
5. Assign a confidence score

Be precise and explain your reasoning clearly."""
    
    def _create_prompt_template(self) -> str:
        """Create the main prompt template for classification."""
        return """<START_OF_SYSTEM_MESSAGE>
{{ system_prompt }}

{% if output_format_str %}
{{ output_format_str }}
{% endif %}

{% if few_shot_demos %}
Here are some examples:

{{ few_shot_demos }}
{% endif %}
<END_OF_SYSTEM_MESSAGE>

<START_OF_USER_MESSAGE>
Text to classify: {{ input_str }}
<END_OF_USER_MESSAGE>"""
    
    def _setup_parameters(self) -> Dict[str, Parameter]:
        """Set up optimizable parameters for the classifier."""
        # Create class information for the task description
        class_info = list(zip(self.class_labels, self.class_descriptions))
        
        task_desc = Prompt(
            template=self.task_template,
            prompt_kwargs={"class_info": class_info}
        )()
        
        return {
            "system_prompt": Parameter(
                data=task_desc,
                role_desc="Task description and classification instructions",
                requires_opt=True,
                param_type=ParameterType.PROMPT
            ),
            "output_format_str": Parameter(
                data=self.parser.get_output_format_str(),
                role_desc="Output format requirements",
                requires_opt=False,
                param_type=ParameterType.PROMPT
            ),
            "few_shot_demos": Parameter(
                data=None,
                requires_opt=True,
                role_desc="Few-shot examples to improve classification",
                param_type=ParameterType.DEMOS
            )
        }
    
    def _prepare_input(self, text: str) -> Dict[str, Parameter]:
        """Prepare input text for the model."""
        input_data = self.input_data_class(text=text)
        input_str = f"'{text}'"
        
        return {
            "input_str": Parameter(
                data=input_str,
                requires_opt=False,
                role_desc="Text to be classified"
            )
        }
    
    def call(
        self,
        text: str,
        id: Optional[str] = None
    ) -> Union[Any, Parameter]:  # Using Any instead of adal.GeneratorOutput
        """
        Classify the given text.
        
        Args:
            text: Input text to classify
            id: Optional ID for tracking
            
        Returns:
            Classification result with reasoning
        """
        prompt_kwargs = self._prepare_input(text)
        output = self.llm(prompt_kwargs=prompt_kwargs, id=id)
        return output
    
    def forward(
        self,
        text: str,
        id: Optional[str] = None
    ) -> Parameter:
        """Forward pass for training mode."""
        return self.call(text, id)


class ClassificationOptimizer(AdalComponent):
    """
    AdalComponent wrapper for prompt optimization of text classification.
    
    This class integrates the TextClassifier with AdalFlow's optimization
    framework, enabling automatic prompt and few-shot optimization.
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict[str, Any],
        class_labels: List[str],
        class_descriptions: Optional[List[str]] = None,
        teacher_model_config: Optional[Dict[str, Any]] = None,
        backward_engine_model_config: Optional[Dict[str, Any]] = None,
        text_optimizer_model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the classification optimizer.
        
        Args:
            model_client: Model client for the main task
            model_kwargs: Model parameters
            class_labels: List of classification labels
            class_descriptions: Optional descriptions for each class
            teacher_model_config: Config for teacher model (few-shot optimization)
            backward_engine_model_config: Config for backward engine
            text_optimizer_model_config: Config for text optimization
        """
        # Initialize the task pipeline
        task = TextClassifier(
            model_client=model_client,
            model_kwargs=model_kwargs,
            class_labels=class_labels,
            class_descriptions=class_descriptions
        )
        
        # Set up evaluation function
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # Set up loss function for optimization
        loss_fn = EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if predicted_label == true_label else 0"
        )
        
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config
        )
        
        self.class_labels = class_labels
    
    def prepare_task(self, sample: ClassificationData) -> Tuple[Callable, Dict[str, Any]]:
        """Prepare the task call for a sample."""
        return self.task.call, {"text": sample.text, "id": getattr(sample, 'id', None)}
    
    def prepare_eval(
        self,
        sample: ClassificationData,
        y_pred: Any  # Using Any instead of adal.GeneratorOutput
    ) -> Tuple[Callable, Dict[str, Any]]:
        """Prepare evaluation for a sample."""
        predicted_label = None
        
        if (y_pred and 
            y_pred.data is not None and 
            hasattr(y_pred.data, 'predicted_label')):
            predicted_label = y_pred.data.predicted_label
        
        return self.eval_fn, {
            "y": predicted_label,
            "y_gt": sample.label
        }
    
    def prepare_loss(
        self,
        sample: ClassificationData,
        y_pred: Parameter,
        *args,
        **kwargs
    ) -> Tuple[Callable, Dict[str, Any]]:
        """Prepare loss calculation for optimization."""
        full_response = y_pred.full_response
        predicted_label = None
        
        if (full_response and 
            full_response.data is not None and 
            hasattr(full_response.data, 'predicted_label')):
            predicted_label = full_response.data.predicted_label
        
        y_pred.eval_input = predicted_label
        
        y_gt = Parameter(
            name="y_gt",
            data=sample.label,
            eval_input=sample.label,
            requires_opt=False
        )
        
        return self.loss_fn, {"kwargs": {"y": y_pred, "y_gt": y_gt}}


class PromptOptimizationPipeline:
    """
    Complete pipeline for prompt optimization in text classification.
    
    This class orchestrates the entire optimization process, from data preparation
    to model training and evaluation.
    """
    
    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        class_labels: List[str] = None,
        class_descriptions: List[str] = None
    ):
        """
        Initialize the optimization pipeline.
        
        Args:
            model_id: AWS Bedrock model identifier
            class_labels: List of classification labels
            class_descriptions: Optional descriptions for each class
        """
        self.model_id = model_id
        self.class_labels = class_labels or ["positive", "negative", "neutral"]
        self.class_descriptions = class_descriptions or self.class_labels
        
        # Initialize model client
        self.model_client = BedrockModelClient(model_id=model_id)
        self.model_kwargs = {"max_tokens": 512}
        
        # Model configurations for optimization
        self.teacher_model_config = {
            "model_client": BedrockModelClient("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "model_kwargs": {"max_tokens": 512}
        }
        
        self.backward_engine_config = {
            "model_client": BedrockModelClient("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "model_kwargs": {"max_tokens": 512}
        }
        
        self.text_optimizer_config = {
            "model_client": BedrockModelClient("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "model_kwargs": {"max_tokens": 1024}
        }
        
        logger.info(f"Initialized optimization pipeline for model: {model_id}")
    
    def create_sample_dataset(self) -> List[ClassificationData]:
        """
        Create a sample dataset for demonstration.
        Replace this with your actual data loading logic.
        """
        sample_data = [
            ClassificationData(text="I love this product! It works perfectly.", label="positive"),
            ClassificationData(text="This is terrible. Complete waste of money.", label="negative"),
            ClassificationData(text="It's okay, nothing special but not bad either.", label="neutral"),
            ClassificationData(text="Amazing quality and fast delivery!", label="positive"),
            ClassificationData(text="Poor customer service, very disappointed.", label="negative"),
            ClassificationData(text="Standard quality, meets expectations.", label="neutral"),
        ]
        return sample_data
    
    def setup_optimizer(self) -> ClassificationOptimizer:
        """Set up the classification optimizer."""
        optimizer = ClassificationOptimizer(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            class_labels=self.class_labels,
            class_descriptions=self.class_descriptions,
            teacher_model_config=self.teacher_model_config,
            backward_engine_model_config=self.backward_engine_config,
            text_optimizer_model_config=self.text_optimizer_config
        )
        
        logger.info("Classification optimizer setup complete")
        return optimizer
    
    def train_and_optimize(
        self,
        train_dataset: List[ClassificationData],
        val_dataset: List[ClassificationData],
        test_dataset: List[ClassificationData] = None,
        max_steps: int = 12,
        train_batch_size: int = 4,
        num_workers: int = 2,
        optimization_order: Literal["sequential", "mixed"] = "sequential",
        raw_shots: int = 0,
        bootstrap_shots: int = 1
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Train and optimize the classification model.
        
        Args:
            train_dataset: Training data
            val_dataset: Validation data  
            test_dataset: Optional test data
            max_steps: Maximum optimization steps
            train_batch_size: Batch size for training
            num_workers: Number of parallel workers
            optimization_order: Order of optimization ("sequential" or "mixed")
            raw_shots: Number of raw examples for few-shot
            bootstrap_shots: Number of bootstrap examples
            
        Returns:
            Tuple of (checkpoint_path, results)
        """
        # Setup the optimizer
        adal_component = self.setup_optimizer()
        
        # Initialize trainer
        trainer = Trainer(
            adaltask=adal_component,
            max_steps=max_steps,
            train_batch_size=train_batch_size,
            num_workers=num_workers,
            optimization_order=optimization_order,
            raw_shots=raw_shots,
            bootstrap_shots=bootstrap_shots,
            strategy="constrained",
            weighted_sampling=True,
            exclude_input_fields_from_bootstrap_demos=True
        )
        
        logger.info(f"Starting training with {len(train_dataset)} training samples...")
        logger.info(f"Optimization order: {optimization_order}")
        logger.info(f"Max steps: {max_steps}, Batch size: {train_batch_size}")
        
        # Run training
        checkpoint_path, results = trainer.fit(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
        
        logger.info(f"Training completed. Checkpoint saved to: {checkpoint_path}")
        return checkpoint_path, results
    
    def evaluate_model(
        self,
        test_dataset: List[ClassificationData],
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test data for evaluation
            checkpoint_path: Path to saved checkpoint (optional)
            
        Returns:
            Evaluation metrics
        """
        # Setup classifier (load from checkpoint if provided)
        if checkpoint_path:
            logger.info(f"Loading model from checkpoint: {checkpoint_path}")
            # In a real implementation, you'd load the optimized parameters here
        
        classifier = TextClassifier(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            class_labels=self.class_labels,
            class_descriptions=self.class_descriptions
        )
        
        correct = 0
        total = len(test_dataset)
        
        logger.info(f"Evaluating on {total} test samples...")
        
        for sample in test_dataset:
            try:
                result = classifier.call(sample.text)
                if (result.data and 
                    hasattr(result.data, 'predicted_label') and
                    result.data.predicted_label == sample.label):
                    correct += 1
            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
        
        accuracy = correct / total if total > 0 else 0.0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        logger.info(f"Evaluation results: {metrics}")
        return metrics
    
    def run_complete_pipeline(
        self,
        train_data: Optional[List[ClassificationData]] = None,
        val_data: Optional[List[ClassificationData]] = None,
        test_data: Optional[List[ClassificationData]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete optimization pipeline.
        
        Args:
            train_data: Training data (uses sample data if None)
            val_data: Validation data (uses sample data if None)  
            test_data: Test data (uses sample data if None)
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete optimization pipeline...")
        
        # Use sample data if none provided
        if not train_data:
            sample_data = self.create_sample_dataset()
            train_data = sample_data[:4]
            val_data = sample_data[4:5] 
            test_data = sample_data[5:]
        
        # Training and optimization
        checkpoint_path, training_results = self.train_and_optimize(
            train_dataset=train_data,
            val_dataset=val_data,
            test_dataset=test_data,
            max_steps=6,  # Reduced for demo
            train_batch_size=2
        )
        
        # Evaluation
        eval_results = self.evaluate_model(test_data, checkpoint_path)
        
        results = {
            "checkpoint_path": checkpoint_path,
            "training_results": training_results,
            "evaluation_results": eval_results,
            "class_labels": self.class_labels
        }
        
        logger.info("Pipeline completed successfully!")
        return results


# Example usage and testing
def main():
    """Main function demonstrating the prompt optimization pipeline."""
    
    # Initialize the pipeline for sentiment analysis
    pipeline = PromptOptimizationPipeline(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        class_labels=["positive", "negative", "neutral"],
        class_descriptions=[
            "Positive sentiment - expresses satisfaction, happiness, or approval",
            "Negative sentiment - expresses dissatisfaction, anger, or disapproval", 
            "Neutral sentiment - balanced or factual without strong emotion"
        ]
    )
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("PROMPT OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Checkpoint Path: {results['checkpoint_path']}")
    print(f"Final Accuracy: {results['evaluation_results']['accuracy']:.2%}")
    print(f"Class Labels: {', '.join(results['class_labels'])}")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Set up environment (adjust as needed)
    os.environ.setdefault("AWS_REGION", "us-east-1")
    
    # Run the main pipeline
    results = main()