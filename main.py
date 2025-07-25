import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Configure page layout
st.set_page_config(
    page_title="AWS Knowledge Bases RAG Chat",
    layout="wide",
)

class AWSKnowledgeBasesParser:
    """Parser for AWS Knowledge Bases Retrieve and Generate output"""
    
    @staticmethod
    def parse_kb_response(kb_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse AWS Knowledge Bases response into a format suitable for Streamlit
        
        Args:
            kb_response: Raw response from AWS Knowledge Bases
            
        Returns:
            Parsed response with citations and references
        """
        try:
            # Extract the main response text
            response_text = kb_response.get("output", {}).get("text", "")
            
            # Parse citations to create references and citation mappings
            citations = kb_response.get("citations", [])
            references = []
            citation_spans = []
            
            for citation in citations:
                # Extract citation span information
                generated_part = citation.get("generatedResponsePart", {})
                text_part = generated_part.get("textResponsePart", {})
                span = text_part.get("span", {})
                cited_text = text_part.get("text", "")
                
                citation_spans.append({
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                    "text": cited_text
                })
                
                # Extract references for this citation
                retrieved_refs = citation.get("retrievedReferences", [])
                for ref in retrieved_refs:
                    ref_content = ref.get("content", {})
                    ref_location = ref.get("location", {})
                    
                    reference = {
                        "text": ref_content.get("text", "")[:100] + "..." if len(ref_content.get("text", "")) > 100 else ref_content.get("text", ""),
                        "full_text": ref_content.get("text", ""),
                        "url": AWSKnowledgeBasesParser._extract_s3_uri(ref_location),
                        "location_type": ref_location.get("type", ""),
                        "citation_span": {
                            "start": span.get("start", 0),
                            "end": span.get("end", 0),
                            "cited_text": cited_text
                        }
                    }
                    references.append(reference)
            
            return {
                "response": response_text,
                "references": references,
                "citation_spans": citation_spans,
                "raw_response": kb_response
            }
            
        except Exception as e:
            st.error(f"Error parsing Knowledge Bases response: {str(e)}")
            return {
                "response": "Error parsing response",
                "references": [],
                "citation_spans": [],
                "raw_response": kb_response
            }
    
    @staticmethod
    def _extract_s3_uri(location: Dict[str, Any]) -> str:
        """Extract S3 URI from location object"""
        if location.get("type") == "S3":
            s3_location = location.get("s3Location", {})
            return s3_location.get("uri", "")
        return ""
    
    @staticmethod
    def create_citation_mapping(response_text: str, citation_spans: List[Dict]) -> List[Tuple[int, int, int]]:
        """
        Create mapping of citation numbers to text spans
        
        Returns:
            List of tuples (citation_number, start_pos, end_pos)
        """
        mappings = []
        for i, span in enumerate(citation_spans):
            mappings.append((i + 1, span["start"], span["end"]))
        return mappings

class DocumentManager:
    """Handles document loading and text highlighting"""
    
    @staticmethod
    def load_document_from_reference(reference: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load document content from reference
        In a real implementation, this would fetch from S3 or other storage
        """
        # For demo purposes, return the reference text as document content
        # In production, you would fetch the full document from S3
        return {
            "title": f"Document: {reference['url'].split('/')[-1] if reference['url'] else 'Unknown'}",
            "content": reference.get("full_text", "Document content not available"),
            "url": reference.get("url", ""),
            "citation_span": reference.get("citation_span", {})
        }
    
    @staticmethod
    def highlight_text_in_document(document_content: str, citation_span: Dict[str, Any]) -> str:
        """
        Highlight text in document based on citation span
        """
        try:
            cited_text = citation_span.get("cited_text", "")
            if not cited_text:
                return document_content
            
            # Simple text highlighting - in production you might want more sophisticated matching
            import re
            escaped_text = re.escape(cited_text)
            highlighted_content = re.sub(
                f"({escaped_text})", 
                r"<mark style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: bold;'>\1</mark>", 
                document_content, 
                flags=re.IGNORECASE
            )
            return highlighted_content
            
        except Exception as e:
            st.error(f"Error highlighting text: {str(e)}")
            return document_content

class ChatInterface:
    """Handles chat interface and message rendering"""
    
    @staticmethod
    def render_response_with_citations(response_text: str, citation_spans: List[Dict]) -> str:
        """
        Render response text with citation markers
        """
        if not citation_spans:
            return response_text
        
        # Sort spans by start position in reverse order to avoid position shifts
        sorted_spans = sorted(citation_spans, key=lambda x: x["start"], reverse=True)
        
        result_text = response_text
        for i, span in enumerate(sorted_spans):
            start = span["start"]
            end = span["end"]
            citation_num = len(citation_spans) - i  # Reverse numbering due to reverse sort
            
            # Insert citation marker
            citation_marker = f" **[{citation_num}]**"
            result_text = result_text[:end] + citation_marker + result_text[end:]
        
        return result_text
    
    @staticmethod
    def render_references(references: List[Dict], message_index: int):
        """Render reference buttons"""
        if not references:
            return
        
        st.write("**References:**")
        for j, ref in enumerate(references):
            ref_text = ref.get("text", "No preview available")
            button_label = f"📄 [{j+1}] {ref_text[:60]}{'...' if len(ref_text) > 60 else ''}"
            
            if st.button(button_label, key=f"ref_{message_index}_{j}", use_container_width=True):
                document = DocumentManager.load_document_from_reference(ref)
                st.session_state.selected_document = document
                st.rerun()

# Mock AWS Knowledge Bases response for demo
MOCK_AWS_KB_RESPONSE = {
    "citations": [
        {
            "generatedResponsePart": {
                "textResponsePart": {
                    "span": {"end": 123, "start": 0},
                    "text": "Machine learning is a subset of artificial intelligence"
                }
            },
            "retrievedReferences": [
                {
                    "content": {
                        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. This revolutionary approach has transformed how we solve complex problems across industries, from healthcare diagnostics to autonomous vehicles. The field combines statistical methods, computational algorithms, and domain expertise to extract meaningful patterns from large datasets.",
                        "type": "TEXT"
                    },
                    "location": {
                        "s3Location": {"uri": "s3://knowledge-base/ml-fundamentals.pdf"},
                        "type": "S3"
                    }
                }
            ]
        },
        {
            "generatedResponsePart": {
                "textResponsePart": {
                    "span": {"end": 245, "start": 124},
                    "text": "Deep learning uses neural networks with multiple layers to process complex patterns"
                }
            },
            "retrievedReferences": [
                {
                    "content": {
                        "text": "Deep learning represents a significant advancement in machine learning, utilizing artificial neural networks with multiple layers (hence 'deep') to model and understand complex patterns in data. Each layer learns increasingly sophisticated features, from simple edges and textures in early layers to complete objects and concepts in deeper layers. This hierarchical feature learning enables deep learning models to achieve remarkable performance in tasks like image recognition, natural language processing, and speech synthesis.",
                        "type": "TEXT"
                    },
                    "location": {
                        "s3Location": {"uri": "s3://knowledge-base/deep-learning-guide.pdf"},
                        "type": "S3"
                    }
                }
            ]
        }
    ],
    "output": {
        "text": "Machine learning is a subset of artificial intelligence that enables computers to learn from data. Deep learning uses neural networks with multiple layers to process complex patterns, making it particularly effective for tasks like computer vision and natural language processing."
    }
}

def simulate_aws_kb_query(query: str) -> Dict[str, Any]:
    """
    Simulate AWS Knowledge Bases query
    In production, this would call the actual AWS Knowledge Bases API
    """
    # Return mock response for demo
    return MOCK_AWS_KB_RESPONSE

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_document" not in st.session_state:
    st.session_state.selected_document = None

# Main layout with two columns
left_col, right_col = st.columns([3, 2])

# Left column - Chat
with left_col:
    st.header("💬 AWS Knowledge Bases RAG Chat")
    
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        if message["type"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                # Display formatted response with citations
                st.write(message["formatted_response"])
                
                # Display reference buttons
                ChatInterface.render_references(message.get("references", []), i)

# Right column - Document Viewer
with right_col:
    st.header("📄 Document Viewer")
    
    if st.session_state.selected_document:
        # Close button
        if st.button("❌ Close Document", type="secondary"):
            st.session_state.selected_document = None
            st.rerun()
        
        # Document info
        doc = st.session_state.selected_document
        st.subheader(doc['title'])
        
        if doc.get('url'):
            st.caption(f"Source: {doc['url']}")
        
        # Show citation info if available
        citation_span = doc.get('citation_span', {})
        if citation_span.get('cited_text'):
            st.info(f"🔍 Highlighted citation: \"{citation_span['cited_text']}\"")
        
        # Document content with highlighting
        content = doc['content']
        if citation_span:
            content = DocumentManager.highlight_text_in_document(content, citation_span)
        
        with st.container():
            st.markdown(content, unsafe_allow_html=True)
    else:
        st.info("Click on a reference button from the chat to view the source document.")

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "type": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Simulate AWS Knowledge Bases query
    with st.spinner("Searching knowledge base..."):
        raw_kb_response = simulate_aws_kb_query(user_input)
        
        # Parse the response
        parsed_response = AWSKnowledgeBasesParser.parse_kb_response(raw_kb_response)
        
        # Format response with citations
        formatted_response = ChatInterface.render_response_with_citations(
            parsed_response["response"], 
            parsed_response["citation_spans"]
        )
        
        # Add assistant response
        st.session_state.messages.append({
            "type": "assistant",
            "content": parsed_response["response"],
            "formatted_response": formatted_response,
            "references": parsed_response["references"],
            "citation_spans": parsed_response["citation_spans"],
            "timestamp": timestamp
        })
    
    st.rerun()