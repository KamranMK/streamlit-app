import streamlit as st
import json
from datetime import datetime

# Configure page layout
st.set_page_config(
    page_title="RAG Chat Interface",
    layout="wide",
    # initial_sidebar_state="expanded"
)

# Mock data for RAG responses
MOCK_RESPONSES = [
    {
        "query": "What is machine learning?",
        "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task [1]. This approach differs from traditional programming by allowing systems to improve their performance through experience and data analysis [2]. The field has evolved significantly since its inception in the 1950s [3], with modern applications ranging from healthcare diagnostics [1] to autonomous vehicles [4].",
        "references": [
            {
                "text": "Machine Learning (ML) is a revolutionary field of computer science that has transformed how we approach problem-solving in the digital age.",
                "url": "s3://documents/ml-intro-ch1.pdf"
            },
            {
                "text": "Data-Driven Decision Making: Unlike traditional programming where rules are explicitly coded, ML systems derive rules from data patterns.",
                "url": "s3://documents/ai-fundamentals-2-3.pdf"
            },
            {
                "text": "The concept of machines that could learn dates back to the 1950s, with pioneers like Alan Turing and Arthur Samuel laying theoretical groundwork.",
                "url": "s3://documents/ai-history.pdf"
            },
            {
                "text": "Self-driving cars represent one of the most ambitious applications of machine learning in transportation.",
                "url": "s3://documents/ml-transportation.pdf"
            }
        ]
    },
    {
        "query": "How does deep learning work?",
        "response": "Deep learning uses neural networks with multiple layers to process data [1]. Each layer learns increasingly complex features, allowing the model to understand patterns in data like images, text, or speech through hierarchical feature learning [2]. The training process involves backpropagation algorithms [1] that adjust weights based on prediction errors [3]. Modern architectures like transformers [4] have revolutionized natural language processing, while convolutional networks [1] remain dominant for computer vision tasks [5].",
        "references": [
            {
                "text": "Deep learning represents a significant advancement in machine learning, utilizing artificial neural networks with multiple layers (hence \"deep\") to model and understand complex patterns in data.",
                "url": "s3://documents/deep-learning-arch.pdf"
            },
            {
                "text": "Artificial neural networks are loosely inspired by biological neurons in the brain, though they are much simpler mathematical models.",
                "url": "s3://documents/neural-networks.pdf"
            },
            {
                "text": "Backpropagation is the cornerstone algorithm for training neural networks. It works by propagating errors backward through the network to update weights.",
                "url": "s3://documents/training-methods.pdf"
            },
            {
                "text": "The Transformer architecture, introduced in \"Attention Is All You Need,\" revolutionized natural language processing and became the foundation for models like GPT and BERT.",
                "url": "s3://documents/transformers.pdf"
            },
            {
                "text": "Convolutional Neural Networks (CNNs) are specifically designed for processing grid-like data such as images.",
                "url": "s3://documents/cnn-vision.pdf"
            }
        ]
    },
    {
        "query": "What are the challenges in AI?",
        "response": "AI faces several significant challenges including bias in algorithms [1], lack of interpretability in complex models [2], and ethical concerns around privacy [3]. Data quality issues can severely impact model performance [1], while computational requirements continue to grow [4]. Regulatory frameworks are still developing [3], and there's an ongoing debate about AI safety and alignment [5]. Additionally, the AI talent shortage [6] and energy consumption of large models [4] present practical implementation challenges.",
        "references": [
            {
                "text": "Historical Bias: When training data reflects past societal inequalities.",
                "url": "s3://documents/ai-bias.pdf"
            },
            {
                "text": "Many modern AI systems, particularly deep learning models, operate as \"black boxes\" where the decision-making process is opaque.",
                "url": "s3://documents/explainable-ai.pdf"
            },
            {
                "text": "Privacy Concerns: Data Collection: How personal data is gathered and used for AI training",
                "url": "s3://documents/ai-ethics.pdf"
            },
            {
                "text": "Modern AI models require enormous computational resources, with training costs reaching millions of dollars for large language models.",
                "url": "s3://documents/ai-compute.pdf"
            },
            {
                "text": "Ensuring that AI systems pursue intended goals and behave as expected, especially as they become more powerful and autonomous.",
                "url": "s3://documents/ai-safety.pdf"
            },
            {
                "text": "There is a significant shortage of qualified AI professionals worldwide, with demand far exceeding supply.",
                "url": "s3://documents/ai-talent.pdf"
            }
        ]
    }
]

# Mock document content
MOCK_DOCUMENTS = {
    "s3://documents/ml-intro-ch1.pdf": {
        "title": "Introduction to Machine Learning - Chapter 1",
        "content": """
# Introduction to Machine Learning

Machine Learning (ML) is a revolutionary field of computer science that has transformed how we approach problem-solving in the digital age. At its core, ML is about creating systems that can learn from data and make predictions or decisions without being explicitly programmed for each specific task.

## Key Concepts

**Supervised Learning**: This approach uses labeled training data to learn a mapping from inputs to outputs. Common examples include email spam detection and image classification.

**Unsupervised Learning**: Here, algorithms find hidden patterns in data without labeled examples. Clustering customer segments and dimensionality reduction are typical applications.

**Reinforcement Learning**: This paradigm involves learning through interaction with an environment, receiving rewards or penalties for actions taken. It's the foundation of game-playing AI and autonomous systems.

## Applications

Machine learning has found applications across virtually every industry:
- Healthcare: Diagnostic imaging and drug discovery
- Finance: Fraud detection and algorithmic trading
- Technology: Recommendation systems and natural language processing
- Transportation: Autonomous vehicles and route optimization

The field continues to evolve rapidly, with new techniques and applications emerging regularly.
        """
    },
    "s3://documents/ai-fundamentals-2-3.pdf": {
        "title": "AI Fundamentals - Section 2.3",
        "content": """
# AI Fundamentals - Section 2.3: Machine Learning Foundations

## Historical Context

The concept of machines that could learn dates back to the 1950s, with pioneers like Alan Turing and Arthur Samuel laying theoretical groundwork. However, practical applications remained limited until the advent of big data and powerful computing resources.

## Core Principles

Machine learning operates on several fundamental principles:

1. **Data-Driven Decision Making**: Unlike traditional programming where rules are explicitly coded, ML systems derive rules from data patterns.

2. **Generalization**: The ability to perform well on unseen data, not just the training examples.

3. **Feature Learning**: Identifying which aspects of the input data are most relevant for making predictions.

## Mathematical Foundations

ML heavily relies on statistics, linear algebra, and calculus. Key mathematical concepts include:
- Probability distributions and Bayes' theorem
- Linear algebra for data representation
- Optimization techniques for model training
- Information theory for measuring uncertainty

## Model Evaluation

Proper evaluation is crucial for ML success. Common metrics include accuracy, precision, recall, and F1-score for classification tasks, while regression problems use metrics like mean squared error and R-squared.
        """
    },
    "s3://documents/deep-learning-arch.pdf": {
        "title": "Deep Learning Architecture Guide",
        "content": """
# Deep Learning Architecture Guide

## Neural Network Fundamentals

Deep learning represents a significant advancement in machine learning, utilizing artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data.

## Architecture Components

**Input Layer**: Receives raw data and passes it to the network. The size depends on the dimensionality of your input data.

**Hidden Layers**: These intermediate layers perform the bulk of the computation. Each layer transforms the input using weighted connections and activation functions.

**Output Layer**: Produces the final prediction or classification. The structure depends on the specific task (regression vs classification).

## Common Architectures

**Feedforward Networks**: Information flows in one direction from input to output. Suitable for basic classification and regression tasks.

**Convolutional Neural Networks (CNNs)**: Specialized for processing grid-like data such as images. They use convolution operations to detect local features.

**Recurrent Neural Networks (RNNs)**: Designed for sequential data like text or time series. They maintain memory of previous inputs through hidden states.

**Transformer Architecture**: The current state-of-the-art for many NLP tasks, using attention mechanisms to process sequences in parallel.

## Training Process

Deep learning models are trained using backpropagation, which calculates gradients and updates weights to minimize prediction errors. This process requires careful tuning of hyperparameters like learning rate, batch size, and network architecture.

## Challenges and Solutions

- **Overfitting**: Use regularization techniques like dropout and batch normalization
- **Vanishing Gradients**: Employ residual connections and proper weight initialization
- **Computational Cost**: Utilize GPUs and distributed training strategies
        """
    },
    "s3://documents/ai-history.pdf": {
        "title": "History of AI and Machine Learning",
        "content": """
# History of AI and Machine Learning

## Early Beginnings (1950s-1960s)

The field of artificial intelligence was formally founded in 1956 at the Dartmouth Conference. Key pioneers included Alan Turing, who proposed the famous Turing Test, and Arthur Samuel, who coined the term "machine learning."

## The AI Winters (1970s-1980s)

Due to overpromising and underdelivering, AI research faced significant funding cuts during periods known as "AI winters." Progress was slower than expected, leading to skepticism about the field's potential.

## Revival and Expert Systems (1980s-1990s)

AI experienced a resurgence with the development of expert systems that could mimic human decision-making in specific domains. Companies began investing heavily in AI applications.

## The Rise of Machine Learning (1990s-2000s)

Statistical approaches to machine learning gained prominence, with algorithms like support vector machines and random forests becoming popular for practical applications.

## Deep Learning Revolution (2010s-Present)

The availability of big data and powerful GPUs enabled the deep learning revolution, leading to breakthroughs in computer vision, natural language processing, and game playing.
        """
    },
    "s3://documents/ml-transportation.pdf": {
        "title": "ML Applications in Transportation",
        "content": """
# Machine Learning Applications in Transportation

## Autonomous Vehicles

Self-driving cars represent one of the most ambitious applications of machine learning in transportation. These systems use computer vision, sensor fusion, and decision-making algorithms to navigate safely.

## Traffic Optimization

ML algorithms analyze traffic patterns to optimize signal timing, reduce congestion, and improve overall traffic flow in urban environments.

## Predictive Maintenance

Transportation companies use machine learning to predict when vehicles will need maintenance, reducing downtime and improving safety.

## Route Optimization

Delivery companies and ride-sharing services use ML to find optimal routes, considering factors like traffic, weather, and demand patterns.

## Safety Systems

Advanced driver assistance systems (ADAS) use machine learning to detect potential hazards and assist drivers in avoiding accidents.
        """
    },
    "s3://documents/neural-networks.pdf": {
        "title": "Neural Network Fundamentals",
        "content": """
# Neural Network Fundamentals

## Biological Inspiration

Artificial neural networks are loosely inspired by biological neurons in the brain, though they are much simpler mathematical models.

## Basic Components

**Neurons (Nodes)**: The basic processing units that receive inputs, apply weights, and produce outputs.

**Weights**: Parameters that determine the strength of connections between neurons.

**Activation Functions**: Mathematical functions that introduce non-linearity into the network.

## Learning Process

Neural networks learn through a process of adjusting weights based on training data, typically using gradient descent optimization.

## Types of Networks

- **Feedforward Networks**: Information flows in one direction
- **Recurrent Networks**: Allow feedback loops and memory
- **Convolutional Networks**: Specialized for processing grid-like data
        """
    },
    "s3://documents/training-methods.pdf": {
        "title": "Backpropagation and Training Methods",
        "content": """
# Backpropagation and Training Methods

## Backpropagation Algorithm

Backpropagation is the cornerstone algorithm for training neural networks. It works by propagating errors backward through the network to update weights.

## Gradient Descent Variants

- **Stochastic Gradient Descent (SGD)**: Updates weights after each training example
- **Mini-batch Gradient Descent**: Updates weights after processing small batches
- **Adam Optimizer**: Adaptive learning rate method with momentum

## Regularization Techniques

- **Dropout**: Randomly disables neurons during training to prevent overfitting
- **Batch Normalization**: Normalizes inputs to each layer
- **Weight Decay**: Penalizes large weights to encourage simpler models

## Training Best Practices

Proper initialization, learning rate scheduling, and early stopping are crucial for successful neural network training.
        """
    },
    "s3://documents/transformers.pdf": {
        "title": "Transformer Architecture Explained",
        "content": """
# Transformer Architecture Explained

## Introduction

The Transformer architecture, introduced in "Attention Is All You Need," revolutionized natural language processing and became the foundation for models like GPT and BERT.

## Key Components

**Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence.

**Multi-Head Attention**: Multiple attention mechanisms running in parallel to capture different types of relationships.

**Position Encoding**: Since transformers don't inherently understand sequence order, position encodings are added to input embeddings.

## Advantages

- **Parallelization**: Unlike RNNs, transformers can process sequences in parallel
- **Long-Range Dependencies**: Better at capturing relationships between distant words
- **Scalability**: Can be scaled to very large sizes effectively

## Applications

Transformers have been successfully applied to translation, text generation, question answering, and even computer vision tasks.
        """
    },
    "s3://documents/cnn-vision.pdf": {
        "title": "Computer Vision with CNNs",
        "content": """
# Computer Vision with Convolutional Neural Networks

## CNN Architecture

Convolutional Neural Networks (CNNs) are specifically designed for processing grid-like data such as images.

## Key Layers

**Convolutional Layers**: Apply filters to detect features like edges, textures, and patterns.

**Pooling Layers**: Reduce spatial dimensions while retaining important information.

**Fully Connected Layers**: Traditional neural network layers that make final classifications.

## Feature Hierarchy

CNNs learn hierarchical features:
- **Low-level**: Edges, corners, basic shapes
- **Mid-level**: Textures, patterns, object parts
- **High-level**: Complete objects and scenes

## Applications

- **Image Classification**: Identifying objects in images
- **Object Detection**: Locating and classifying multiple objects
- **Semantic Segmentation**: Pixel-level classification
- **Medical Imaging**: Diagnostic assistance in healthcare
        """
    },
    "s3://documents/ai-bias.pdf": {
        "title": "AI Bias and Fairness Study",
        "content": """
# AI Bias and Fairness Study

## Types of Bias

**Historical Bias**: When training data reflects past societal inequalities.

**Representation Bias**: When certain groups are underrepresented in training data.

**Algorithmic Bias**: When the algorithm itself introduces unfair treatment.

## Impact Areas

AI bias can affect hiring decisions, loan approvals, criminal justice, healthcare, and many other critical areas of society.

## Mitigation Strategies

- **Diverse Training Data**: Ensuring representative datasets
- **Fairness Metrics**: Measuring and monitoring bias in model outputs
- **Algorithmic Auditing**: Regular assessment of model fairness
- **Inclusive Development Teams**: Diverse perspectives in AI development

## Ongoing Challenges

Defining fairness is complex and context-dependent, making bias mitigation an ongoing challenge in AI development.
        """
    },
    "s3://documents/explainable-ai.pdf": {
        "title": "Explainable AI Research",
        "content": """
# Explainable AI Research

## The Black Box Problem

Many modern AI systems, particularly deep learning models, operate as "black boxes" where the decision-making process is opaque.

## Importance of Explainability

- **Trust**: Users need to understand why AI makes certain decisions
- **Debugging**: Developers need to identify and fix model errors
- **Compliance**: Regulations may require explainable decisions
- **Fairness**: Understanding bias requires insight into decision processes

## Explainability Techniques

**LIME (Local Interpretable Model-Agnostic Explanations)**: Explains individual predictions by learning locally interpretable models.

**SHAP (SHapley Additive exPlanations)**: Assigns importance values to features based on game theory.

**Attention Visualization**: For neural networks, visualizing attention weights can show what the model focuses on.

## Trade-offs

There's often a trade-off between model performance and explainability, requiring careful consideration based on the application context.
        """
    },
    "s3://documents/ai-ethics.pdf": {
        "title": "AI Ethics and Privacy Report",
        "content": """
# AI Ethics and Privacy Report

## Core Ethical Principles

**Autonomy**: Respecting human agency and decision-making authority.

**Beneficence**: Ensuring AI systems promote human welfare and well-being.

**Non-maleficence**: Preventing harm from AI systems.

**Justice**: Ensuring fair distribution of AI benefits and risks.

## Privacy Concerns

- **Data Collection**: How personal data is gathered and used for AI training
- **Surveillance**: Potential for AI to enable mass surveillance
- **Consent**: Ensuring informed consent for data use
- **Data Ownership**: Questions about who owns and controls personal data

## Regulatory Landscape

Different countries are developing varying approaches to AI regulation, from the EU's comprehensive framework to more sector-specific approaches elsewhere.

## Future Considerations

As AI becomes more powerful and pervasive, ethical considerations will become increasingly important for society.
        """
    },
    "s3://documents/ai-compute.pdf": {
        "title": "Computational Resources in AI",
        "content": """
# Computational Resources in AI

## Growing Computational Demands

Modern AI models require enormous computational resources, with training costs reaching millions of dollars for large language models.

## Hardware Requirements

**GPUs**: Graphics processing units are essential for parallel processing in deep learning.

**TPUs**: Google's Tensor Processing Units are specialized for AI workloads.

**Quantum Computing**: Emerging technology that may revolutionize certain AI applications.

## Energy Consumption

Training large AI models consumes significant energy, raising environmental concerns about AI's carbon footprint.

## Cloud Computing

Major cloud providers offer AI-specific services, making advanced AI accessible to smaller organizations.

## Future Trends

- **Model Efficiency**: Research into more efficient architectures
- **Edge Computing**: Moving AI processing closer to data sources
- **Specialized Hardware**: Custom chips designed for specific AI tasks
        """
    },
    "s3://documents/ai-safety.pdf": {
        "title": "AI Safety and Alignment",
        "content": """
# AI Safety and Alignment

## The Alignment Problem

Ensuring that AI systems pursue intended goals and behave as expected, especially as they become more powerful and autonomous.

## Safety Challenges

**Specification**: Difficulty in precisely specifying what we want AI systems to do.

**Robustness**: Ensuring AI systems work safely in unexpected situations.

**Monitoring**: Detecting when AI systems are behaving unexpectedly.

## Research Areas

- **Value Learning**: Teaching AI systems human values
- **Interpretability**: Understanding how AI systems make decisions
- **Robustness**: Making AI systems reliable under diverse conditions
- **Verification**: Formally proving that AI systems will behave safely

## Long-term Considerations

As AI capabilities advance, ensuring safety and alignment becomes increasingly critical for preventing potential risks.
        """
    },
    "s3://documents/ai-talent.pdf": {
        "title": "AI Talent Gap Analysis",
        "content": """
# AI Talent Gap Analysis

## Current Situation

There is a significant shortage of qualified AI professionals worldwide, with demand far exceeding supply.

## Skills in Demand

- **Machine Learning Engineering**: Building and deploying ML systems
- **Data Science**: Extracting insights from data
- **AI Research**: Advancing the field through novel techniques
- **AI Ethics**: Ensuring responsible AI development

## Educational Initiatives

Universities and online platforms are expanding AI education, but the gap remains substantial.

## Industry Response

- **Training Programs**: Companies investing in employee AI education
- **Competitive Salaries**: High compensation to attract talent
- **Remote Work**: Expanding geographic reach for talent acquisition

## Future Outlook

The talent gap is expected to persist as AI adoption accelerates across industries.
        """
    }
}

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None
if "highlight_text" not in st.session_state:
    st.session_state.highlight_text = None

def simulate_rag_response(query):
    """Simulate RAG response - in real implementation, this would call your RAG system"""
    # Simple keyword matching for demo
    query_lower = query.lower()
    
    if "machine learning" in query_lower or "ml" in query_lower:
        return MOCK_RESPONSES[0]
    elif "deep learning" in query_lower or "neural" in query_lower:
        return MOCK_RESPONSES[1]
    elif "challenge" in query_lower or "problem" in query_lower or "issue" in query_lower:
        return MOCK_RESPONSES[2]
    else:
        # Generic response
        return {
            "query": query,
            "response": f"I understand you're asking about '{query}' [1]. Based on the available documents, I can help you find relevant information.",
            "references": [
                {
                    "text": "General AI Reference",
                    "url": "s3://documents/ml-intro-ch1.pdf"
                }
            ]
        }

def load_document(url, highlight_text=None):
    """Load document content from S3 URL (mocked)"""
    document = MOCK_DOCUMENTS.get(url, {
        "title": "Document Not Found",
        "content": "The requested document could not be loaded."
    })
    
    # If there's text to highlight, add highlighting
    if highlight_text and document["content"]:
        # Escape special characters for regex
        import re
        escaped_text = re.escape(highlight_text)
        # Replace the text with highlighted version
        highlighted_content = re.sub(
            f"({escaped_text})", 
            r"<mark style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;'>\1</mark>", 
            document["content"], 
            flags=re.IGNORECASE
        )
        document = document.copy()
        document["content"] = highlighted_content
    
    return document

def render_response_with_references(response_text, references, message_index):
    """Render response text with clickable reference numbers"""
    import re
    
    # Find all reference numbers in the text
    ref_pattern = r'\[(\d+)\]'
    matches = list(re.finditer(ref_pattern, response_text))
    
    if not matches:
        return response_text
    
    # Split text and create clickable references
    result_parts = []
    last_end = 0
    
    for match in matches:
        # Add text before the reference
        result_parts.append(response_text[last_end:match.start()])
        
        # Get reference number
        ref_num = int(match.group(1))
        
        if ref_num <= len(references):
            # Create markdown link that will be handled by Streamlit
            result_parts.append(f"**[{ref_num}]**")
        else:
            result_parts.append(match.group(0))
        
        last_end = match.end()
    
    # Add remaining text
    result_parts.append(response_text[last_end:])
    
    return "".join(result_parts)

# Main layout with two columns
left_col, right_col = st.columns([3, 2])

# Left column - Chat
with left_col:
    st.header("💬 Chat Messages & History")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["type"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                # Display response with formatted references
                formatted_response = render_response_with_references(
                    message["content"], 
                    message.get("references", []), 
                    i
                )
                st.write(formatted_response)
                
                # Display reference buttons
                if "references" in message and message["references"]:
                    st.write("**References:**")
                    cols = st.columns(len(message["references"]))
                    for j, ref in enumerate(message["references"]):
                        with cols[j]:
                            if st.button(f"📄 [{j+1}] {ref['text'][:50]}{'...' if len(ref['text']) > 50 else ''}", key=f"ref_{i}_{j}", use_container_width=True):
                                st.session_state.selected_document = load_document(ref['url'], ref['text'])
                                st.session_state.highlight_text = ref['text']
                                st.rerun()

# Right column - Document Viewer
with right_col:
    st.header("📄 Document Viewer")
    
    if st.session_state.selected_document:
        # Close button at the top
        if st.button("❌ Close Document", type="secondary"):
            st.session_state.selected_document = None
            st.session_state.highlight_text = None
            st.rerun()
        
        # Document title
        st.subheader(st.session_state.selected_document['title'])
        
        # Show highlighted text info if available
        if st.session_state.highlight_text:
            st.info(f"🔍 Highlighting: \"{st.session_state.highlight_text}\"")
        
        # Document content in a scrollable container with highlighting
        with st.container():
            st.markdown(st.session_state.selected_document['content'], unsafe_allow_html=True)
    else:
        st.info("Click on a reference button from the chat to view document content here.")

# Chat input at the bottom
user_input = st.chat_input("Ask me anything about the documents...")

if user_input:
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "type": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Simulate RAG response
    rag_response = simulate_rag_response(user_input)
    
    # Add assistant response
    st.session_state.messages.append({
        "type": "assistant",
        "content": rag_response["response"],
        "references": rag_response["references"],
        "timestamp": timestamp
    })
    
    st.rerun()