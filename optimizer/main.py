"""
AdalFlow + AWS Bedrock Text Classification with Prompt Optimization

This module provides a complete implementation for text classification using:
- AdalFlow for prompt optimization and workflow management
- AWS Bedrock for model inference
- Instructor-style structured output using AdalFlow's DataClass
- Comprehensive prompt optimization capabilities

Requirements:
    pip install adalflow boto3 pydantic python-dotenv

Environment Variables:
    AWS_REGION=us-east-1
    AWS_ACCESS_KEY_ID=your_access_key
    AWS_SECRET_ACCESS_KEY=your_secret_key
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union, Any, Tuple, Callable
from pydantic import BaseModel

import boto3
import adalflow as adal
from adalflow.core.model_client import ModelClient
from adalflow.core.types import ModelType, GeneratorOutput
from adalflow.optim.parameter import ParameterType
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BedrockClient(ModelClient):
    """
    Custom AWS Bedrock client for AdalFlow integration.
    
    This client provides seamless integration between AdalFlow and AWS Bedrock,
    enabling the use of various foundation models like Claude, Titan, etc.
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        **kwargs
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region: AWS region for Bedrock service
            model_id: Bedrock model identifier
            **kwargs: Additional parameters for model calls
        """
        super().__init__()
        self.region = region
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_kwargs = kwargs
        
    def init_sync_client(self):
        """Initialize synchronous client."""
        return self.client
    
    def call(self, api_kwargs: Dict, model_type: ModelType = ModelType.LLM) -> Any:
        """
        Make a call to AWS Bedrock.
        
        Args:
            api_kwargs: API arguments including prompt and model parameters
            model_type: Type of model being called
            
        Returns:
            Raw response from Bedrock
        """
        try:
            # Extract prompt from api_kwargs
            prompt = api_kwargs.get("prompt", "")
            
            # Prepare request body for Claude models
            if "anthropic.claude" in self.model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": api_kwargs.get("max_tokens", 1000),
                    "temperature": api_kwargs.get("temperature", 0.0),
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
            else:
                # Generic body structure for other models
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": api_kwargs.get("max_tokens", 1000),
                        "temperature": api_kwargs.get("temperature", 0.0),
                    }
                }
            
            # Make the API call
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=str(body).encode('utf-8'),
                contentType="application/json"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Bedrock API call failed: {e}")
            raise
    
    def parse_chat_completion(self, completion: Any) -> Any:
        """
        Parse the completion response from Bedrock.
        
        Args:
            completion: Raw response from Bedrock
            
        Returns:
            Parsed response with text content
        """
        import json
        
        try:
            # Read response body
            response_body = json.loads(completion['body'].read())
            
            if "anthropic.claude" in self.model_id:
                # Parse Claude response
                content = response_body.get("content", [])
                if content and isinstance(content, list):
                    text = content[0].get("text", "")
                else:
                    text = response_body.get("completion", "")
            else:
                # Parse other model responses
                results = response_body.get("results", [])
                if results:
                    text = results[0].get("outputText", "")
                else:
                    text = response_body.get("completion", "")
            
            # Return in AdalFlow expected format
            return type('Response', (), {
                'choices': [type('Choice', (), {
                    'message': type('Message', (), {'content': text})()
                })()]
            })()
            
        except Exception as e:
            logger.error(f"Failed to parse Bedrock response: {e}")
            raise


@dataclass
class ClassificationData(adal.DataClass):
    """
    Structured data class for text classification with reasoning.
    
    This follows the Instructor pattern for structured outputs but uses
    AdalFlow's DataClass for seamless integration with the optimization framework.
    """
    
    text: str = field(
        metadata={"desc": "The text to be classified"}, 
        default=None
    )
    reasoning: str = field(
        metadata={"desc": "Step-by-step reasoning for the classification decision"}, 
        default=None
    )
    confidence: float = field(
        metadata={"desc": "Confidence score between 0.0 and 1.0"}, 
        default=None
    )
    category: Literal["positive", "negative", "neutral"] = field(
        metadata={"desc": "The predicted category"}, 
        default=None
    )
    
    # Define input and output fields for AdalFlow
    __input_fields__ = ["text"]
    __output_fields__ = ["reasoning", "confidence", "category"]


class TextClassifier(adal.Component):
    """
    Text classification component with structured output.
    
    This component implements a complete text classification pipeline with:
    - Customizable prompts for different classification tasks
    - Structured output parsing
    - Built-in reasoning capabilities
    - Optimization support through Parameter system
    """
    
    def __init__(
        self, 
        model_client: ModelClient, 
        model_kwargs: Dict,
        categories: List[str] = None,
        task_description: str = None
    ):
        """
        Initialize the text classifier.
        
        Args:
            model_client: The model client (e.g., BedrockClient)
            model_kwargs: Model-specific parameters
            categories: List of classification categories
            task_description: Custom task description
        """
        super().__init__()
        
        # Default categories and description
        self.categories = categories or ["positive", "negative", "neutral"]
        self.task_description = task_description or "sentiment analysis"
        
        # Create dynamic template for different classification tasks
        self.template = self._create_template()
        
        # Set up the system prompt as an optimizable parameter
        system_prompt = adal.Parameter(
            data=self._create_system_prompt(),
            role_desc="System prompt for text classification task",
            requires_opt=True,  # Enable optimization
            param_type=ParameterType.PROMPT,
        )
        
        # Set up the data parser for structured output
        self.parser = adal.DataClassParser(
            data_class=ClassificationData, 
            return_data_class=True,
            format_type="yaml"  # Use YAML format for better readability
        )
        
        # Initialize the generator with all components
        self.generator = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=self.template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "output_format_str": self.parser.get_output_format_str(),
                "categories": self.categories,
                "task_description": self.task_description,
            },
            output_processors=self.parser,
        )
    
    def _create_template(self) -> str:
        """Create the prompt template with proper formatting."""
        return """<START_OF_SYSTEM_PROMPT>
{{system_prompt}}

Task: Perform {{task_description}} on the given text.
Categories: {{categories}}

{{output_format_str}}
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
Text to classify: {{input_text}}
<END_OF_USER>"""
    
    def _create_system_prompt(self) -> str:
        """Create the initial system prompt."""
        return """You are an expert text classifier. Your task is to:

1. Carefully read and analyze the input text
2. Provide step-by-step reasoning for your classification decision
3. Assign a confidence score based on how certain you are
4. Select the most appropriate category from the available options

Be thorough in your reasoning and honest about your confidence level."""
    
    def call(self, input_text: str, id: Optional[str] = None) -> GeneratorOutput:
        """
        Classify the input text.
        
        Args:
            input_text: Text to classify
            id: Optional identifier for tracking
            
        Returns:
            Classification result with structured output
        """
        return self.generator(
            prompt_kwargs={"input_text": input_text}, 
            id=id
        )
    
    def bicall(self, input_text: str, id: Optional[str] = None) -> Union[GeneratorOutput, adal.Parameter]:
        """
        Bidirectional call for training mode.
        
        Args:
            input_text: Text to classify
            id: Optional identifier for tracking
            
        Returns:
            Classification result (Parameter during training, GeneratorOutput during inference)
        """
        return self.generator(
            prompt_kwargs={"input_text": input_text}, 
            id=id
        )


class ClassificationTrainer(adal.AdalComponent):
    """
    Training component for optimizing the text classification pipeline.
    
    This component orchestrates the training process using AdalFlow's optimization
    framework, including both prompt optimization and few-shot learning.
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        model_kwargs: Dict,
        teacher_model_config: Dict,
        backward_engine_model_config: Dict,
        text_optimizer_model_config: Dict,
        categories: List[str] = None,
        task_description: str = None,
    ):
        """
        Initialize the training component.
        
        Args:
            model_client: Primary model client for the task
            model_kwargs: Model parameters
            teacher_model_config: Configuration for the teacher model
            backward_engine_model_config: Configuration for backward optimization
            text_optimizer_model_config: Configuration for text optimization
            categories: Classification categories
            task_description: Description of the classification task
        """
        # Initialize the task component
        task = TextClassifier(
            model_client=model_client,
            model_kwargs=model_kwargs,
            categories=categories,
            task_description=task_description,
        )
        
        # Set up evaluation function
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        
        # Set up loss function for optimization
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="Exact match accuracy: 1 if prediction equals ground truth, 0 otherwise"
        )
        
        # Initialize the parent component
        super().__init__(
            task=task,
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            backward_engine_model_config=backward_engine_model_config,
            text_optimizer_model_config=text_optimizer_model_config,
            teacher_model_config=teacher_model_config,
        )
    
    def prepare_task(self, sample: ClassificationData) -> Tuple[Callable, Dict]:
        """
        Prepare the task for execution.
        
        Args:
            sample: Input sample
            
        Returns:
            Task function and its arguments
        """
        return self.task.bicall, {"input_text": sample.text, "id": sample.id if hasattr(sample, 'id') else None}
    
    def prepare_eval(self, sample: ClassificationData, y_pred: GeneratorOutput) -> Tuple[Callable, Dict]:
        """
        Prepare evaluation step.
        
        Args:
            sample: Input sample
            y_pred: Model prediction
            
        Returns:
            Evaluation function and its arguments
        """
        predicted_category = None
        if y_pred and y_pred.data and hasattr(y_pred.data, 'category'):
            predicted_category = y_pred.data.category
        
        return self.eval_fn, {
            "y": predicted_category, 
            "y_gt": sample.category
        }
    
    def prepare_loss(self, sample: ClassificationData, y_pred: adal.Parameter, *args, **kwargs) -> Tuple[Callable, Dict]:
        """
        Prepare loss computation for optimization.
        
        Args:
            sample: Input sample
            y_pred: Model prediction parameter
            
        Returns:
            Loss function and its arguments
        """
        predicted_category = None
        if y_pred.data and y_pred.data.data and hasattr(y_pred.data.data, 'category'):
            predicted_category = y_pred.data.data.category
        
        # Set evaluation input for optimization
        y_pred.eval_input = predicted_category
        
        # Create ground truth parameter
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.category,
            eval_input=sample.category,
            requires_opt=False,
        )
        
        return self.loss_fn, {
            "kwargs": {"y": y_pred, "y_gt": y_gt},
            "id": sample.id if hasattr(sample, 'id') else None,
        }


class OptimizedTextClassificationPipeline:
    """
    Complete pipeline for text classification with prompt optimization.
    
    This class provides a high-level interface for:
    - Setting up the classification model
    - Training with prompt optimization
    - Evaluating performance
    - Making predictions
    """
    
    def __init__(
        self,
        region: str = "us-east-1",
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        categories: List[str] = None,
        task_description: str = None,
    ):
        """
        Initialize the classification pipeline.
        
        Args:
            region: AWS region for Bedrock
            model_id: Bedrock model identifier
            categories: Classification categories
            task_description: Description of the classification task
        """
        self.region = region
        self.model_id = model_id
        self.categories = categories or ["positive", "negative", "neutral"]
        self.task_description = task_description or "sentiment analysis"
        
        # Initialize model configurations
        self.model_client = BedrockClient(region=region, model_id=model_id)
        self.model_kwargs = {"temperature": 0.0, "max_tokens": 1000}
        
        # Set up optimization model configs (using same model for simplicity)
        self.teacher_model_config = {
            "model_client": BedrockClient(region=region, model_id=model_id),
            "model_kwargs": {"temperature": 0.0, "max_tokens": 1500}
        }
        
        self.backward_engine_model_config = self.teacher_model_config
        self.text_optimizer_model_config = self.teacher_model_config
        
        # Initialize components
        self.classifier = None
        self.trainer = None
        self.trained_checkpoint = None
    
    def setup_classifier(self):
        """Set up the text classifier component."""
        self.classifier = TextClassifier(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            categories=self.categories,
            task_description=self.task_description,
        )
        logger.info("Text classifier initialized successfully")
    
    def setup_trainer(self):
        """Set up the training component for optimization."""
        self.trainer_component = ClassificationTrainer(
            model_client=self.model_client,
            model_kwargs=self.model_kwargs,
            teacher_model_config=self.teacher_model_config,
            backward_engine_model_config=self.backward_engine_model_config,
            text_optimizer_model_config=self.text_optimizer_model_config,
            categories=self.categories,
            task_description=self.task_description,
        )
        
        self.trainer = adal.Trainer(
            adaltask=self.trainer_component,
            max_steps=12,  # Adjust based on your needs
            raw_shots=1,   # Number of raw demonstrations
            bootstrap_shots=1,  # Number of bootstrap demonstrations
        )
        logger.info("Trainer initialized successfully")
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None, test_data: List[Dict] = None):
        """
        Train the model with prompt optimization.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset
            
        Returns:
            Training checkpoint
        """
        if not self.trainer:
            self.setup_trainer()
        
        # Convert data to proper format
        train_samples = [ClassificationData(**item) for item in train_data]
        val_samples = [ClassificationData(**item) for item in val_data] if val_data else None
        test_samples = [ClassificationData(**item) for item in test_data] if test_data else None
        
        logger.info(f"Starting training with {len(train_samples)} samples")
        
        # Run training
        checkpoint, _ = self.trainer(
            train_dataset=train_samples,
            val_dataset=val_samples,
            test_dataset=test_samples,
        )
        
        self.trained_checkpoint = checkpoint
        logger.info("Training completed successfully")
        
        return checkpoint
    
    def predict(self, text: str) -> Dict:
        """
        Make a prediction on input text.
        
        Args:
            text: Input text to classify
            
        Returns:
            Classification result
        """
        if not self.classifier:
            self.setup_classifier()
        
        try:
            result = self.classifier(input_text=text)
            
            if result and result.data:
                return {
                    "text": text,
                    "category": result.data.category,
                    "reasoning": result.data.reasoning,
                    "confidence": result.data.confidence,
                    "raw_output": str(result.data)
                }
            else:
                return {"error": "No valid prediction generated", "text": text}
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e), "text": text}
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """
        Make predictions on multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of classification results
        """
        return [self.predict(text) for text in texts]


def create_sample_data() -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create sample data for demonstration.
    
    Returns:
        Train, validation, and test datasets
    """
    train_data = [
        {"text": "I love this product! It's amazing.", "category": "positive"},
        {"text": "This is the worst thing I've ever bought.", "category": "negative"},
        {"text": "It's okay, nothing special.", "category": "neutral"},
        {"text": "Absolutely fantastic! Highly recommend.", "category": "positive"},
        {"text": "Terrible quality, very disappointed.", "category": "negative"},
        {"text": "Average product, does the job.", "category": "neutral"},
    ]
    
    val_data = [
        {"text": "Pretty good, I'm satisfied.", "category": "positive"},
        {"text": "Not great, could be better.", "category": "negative"},
        {"text": "It's fine, nothing to complain about.", "category": "neutral"},
    ]
    
    test_data = [
        {"text": "Excellent service and quality!", "category": "positive"},
        {"text": "Poor experience, would not recommend.", "category": "negative"},
        {"text": "Standard product, met expectations.", "category": "neutral"},
    ]
    
    return train_data, val_data, test_data


def main():
    """
    Example usage of the optimized text classification pipeline.
    """
    # Set up environment
    os.environ.setdefault("AWS_REGION", "us-east-1")
    
    # Initialize pipeline
    pipeline = OptimizedTextClassificationPipeline(
        region=os.getenv("AWS_REGION", "us-east-1"),
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        categories=["positive", "negative", "neutral"],
        task_description="sentiment analysis",
    )
    
    # Set up classifier for basic usage
    pipeline.setup_classifier()
    
    # Test basic prediction
    test_text = "This product exceeded my expectations!"
    result = pipeline.predict(test_text)
    print(f"Classification result: {result}")
    
    # Optional: Run optimization training
    if input("Run optimization training? (y/n): ").lower().startswith('y'):
        train_data, val_data, test_data = create_sample_data()
        
        try:
            checkpoint = pipeline.train(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )
            print(f"Training completed! Checkpoint saved: {checkpoint}")
            
            # Test optimized model
            optimized_result = pipeline.predict(test_text)
            print(f"Optimized classification result: {optimized_result}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    # Test batch prediction
    batch_texts = [
        "I'm very happy with this purchase!",
        "This is completely useless.",
        "It works as expected."
    ]
    
    batch_results = pipeline.batch_predict(batch_texts)
    print(f"Batch classification results: {batch_results}")


if __name__ == "__main__":
    main()