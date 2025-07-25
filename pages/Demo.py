import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Configure page layout
st.set_page_config(
    page_title="AWS Knowledge Bases RAG Chat",
    layout="wide",
)

class AWSKnowledgeBaseParser:
    """Parser for AWS Knowledge Bases output format"""
    
    @staticmethod
    def parse_aws_response(aws_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse AWS Knowledge Bases response into a format suitable for Streamlit UI
        
        Args:
            aws_response: Raw response from AWS Knowledge Bases API
            
        Returns:
            Parsed response with structured references and content
        """
        try:
            # Extract the main generated text
            output_text = aws_response.get("output", {}).get("text", "")
            
            # Parse citations and references
            citations = aws_response.get("citations", [])
            references = []
            citation_map = {}
            
            for citation in citations:
                # Get the generated response part
                generated_part = citation.get("generatedResponsePart", {})
                text_response = generated_part.get("textResponsePart", {})
                
                # Extract span information for highlighting
                span = text_response.get("span", {})
                citation_text = text_response.get("text", "")
                
                # Process retrieved references
                retrieved_refs = citation.get("retrievedReferences", [])
                
                for ref in retrieved_refs:
                    ref_content = ref.get("content", {})
                    ref_text = ref_content.get("text", "")
                    
                    # Extract location information
                    location = ref.get("location", {})
                    s3_location = location.get("s3Location", {})
                    document_uri = s3_location.get("uri", "")
                    
                    # Create reference entry
                    reference_entry = {
                        "text": ref_text,
                        "url": document_uri,
                        "citation_text": citation_text,
                        "span_start": span.get("start", 0),
                        "span_end": span.get("end", 0)
                    }
                    
                    references.append(reference_entry)
            
            # Create numbered references for display
            numbered_references = []
            for i, ref in enumerate(references, 1):
                numbered_references.append({
                    "number": i,
                    "text": ref["text"],
                    "url": ref["url"],
                    "citation_text": ref["citation_text"]
                })
            
            return {
                "response": output_text,
                "references": numbered_references,
                "raw_citations": citations,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": f"Error parsing AWS response: {str(e)}",
                "references": [],
                "raw_citations": [],
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    def extract_document_content(document_uri: str, reference_text: str = None) -> Dict[str, Any]:
        """
        Extract document content from S3 URI
        In a real implementation, this would fetch from S3
        
        Args:
            document_uri: S3 URI of the document
            reference_text: Text to highlight in the document
            
        Returns:
            Document content with metadata
        """
        # Extract filename from URI for display
        filename = document_uri.split("/")[-1] if document_uri else "Unknown Document"
        
        # In a real implementation, you would:
        # 1. Parse the S3 URI
        # 2. Use boto3 to fetch the document content
        # 3. Handle different file formats (PDF, TXT, etc.)
        
        # Mock implementation for demo
        mock_content = f"""
# Document: {filename}

This is a placeholder for the actual document content that would be fetched from:
{document_uri}

In a real implementation, this would contain the full document text retrieved from S3.
The system would support various file formats and extract text content appropriately.

Referenced text: "{reference_text if reference_text else 'N/A'}"
        """
        
        return {
            "title": filename,
            "content": mock_content,
            "uri": document_uri,
            "highlight_text": reference_text
        }

class StreamlitRAGInterface:
    """Streamlit interface for AWS Knowledge Bases RAG system"""
    
    def __init__(self):
        self.parser = AWSKnowledgeBaseParser()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "selected_document" not in st.session_state:
            st.session_state.selected_document = None
        if "highlight_text" not in st.session_state:
            st.session_state.highlight_text = None
    
    def simulate_aws_knowledge_base_call(self, query: str) -> Dict[str, Any]:
        """
        Simulate AWS Knowledge Bases API call
        In production, replace this with actual AWS SDK calls
        """
        # Mock AWS Knowledge Bases response
        mock_aws_response = {
            "citations": [
                {
                    "generatedResponsePart": {
                        "textResponsePart": {
                            "span": {"end": 150, "start": 0},
                            "text": f"Based on the documents, here's information about {query}"
                        }
                    },
                    "retrievedReferences": [
                        {
                            "content": {
                                "text": f"Relevant information about {query} from the knowledge base. This would contain actual extracted text from your documents that relates to the user's query.",
                                "type": "TEXT"
                            },
                            "location": {
                                "s3Location": {"uri": f"s3://your-bucket/documents/{query.replace(' ', '_')}.pdf"},
                                "type": "S3"
                            }
                        },
                        {
                            "content": {
                                "text": f"Additional context and details about {query}. This represents another relevant passage found in your document corpus.",
                                "type": "TEXT"
                            },
                            "location": {
                                "s3Location": {"uri": f"s3://your-bucket/documents/reference_{query.replace(' ', '_')}.pdf"},
                                "type": "S3"
                            }
                        }
                    ]
                }
            ],
            "output": {
                "text": f"Based on the available documents, {query} involves several key aspects [1][2]. The knowledge base contains comprehensive information that addresses your question with specific details and examples. This response would be generated by AWS Knowledge Bases using your indexed documents."
            }
        }
        
        return mock_aws_response
    
    def render_response_with_references(self, response_text: str, references: List[Dict], message_index: int):
        """Render response text with clickable reference numbers"""
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
                # Create highlighted reference number
                result_parts.append(f"**[{ref_num}]**")
            else:
                result_parts.append(match.group(0))
            
            last_end = match.end()
        
        # Add remaining text
        result_parts.append(response_text[last_end:])
        
        return "".join(result_parts)
    
    def render_chat_interface(self):
        """Render the main chat interface"""
        # Main layout with two columns
        left_col, right_col = st.columns([3, 2])
        
        # Left column - Chat
        with left_col:
            st.header("💬 AWS Knowledge Bases Chat")
            
            # Display chat messages
            for i, message in enumerate(st.session_state.messages):
                if message["type"] == "user":
                    with st.chat_message("user"):
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        # Display response with formatted references
                        formatted_response = self.render_response_with_references(
                            message["content"], 
                            message.get("references", []), 
                            i
                        )
                        st.write(formatted_response)
                        
                        # Display reference buttons
                        if "references" in message and message["references"]:
                            st.write("**References:**")
                            
                            # Create columns for reference buttons
                            num_refs = len(message["references"])
                            cols = st.columns(min(num_refs, 3))  # Max 3 columns
                            
                            for j, ref in enumerate(message["references"]):
                                col_idx = j % 3
                                with cols[col_idx]:
                                    # Truncate text for button display
                                    button_text = ref['text'][:50] + ('...' if len(ref['text']) > 50 else '')
                                    
                                    if st.button(
                                        f"📄 [{ref['number']}] {button_text}", 
                                        key=f"ref_{i}_{j}", 
                                        use_container_width=True
                                    ):
                                        # Load document content
                                        doc_content = self.parser.extract_document_content(
                                            ref['url'], 
                                            ref['text']
                                        )
                                        st.session_state.selected_document = doc_content
                                        st.session_state.highlight_text = ref['text']
                                        st.rerun()
        
        # Right column - Document Viewer
        with right_col:
            self.render_document_viewer()
    
    def render_document_viewer(self):
        """Render the document viewer panel"""
        st.header("📄 Document Viewer")
        
        if st.session_state.selected_document:
            # Close button at the top
            if st.button("❌ Close Document", type="secondary"):
                st.session_state.selected_document = None
                st.session_state.highlight_text = None
                st.rerun()
            
            # Document metadata
            doc = st.session_state.selected_document
            st.subheader(doc['title'])
            
            # Show S3 URI
            with st.expander("📍 Document Location"):
                st.code(doc['uri'])
            
            # Show highlighted text info if available
            if st.session_state.highlight_text:
                st.info(f"🔍 Highlighting: \"{st.session_state.highlight_text[:100]}{'...' if len(st.session_state.highlight_text) > 100 else ''}\"")
            
            # Document content in a scrollable container
            with st.container():
                content = doc['content']
                
                # Apply highlighting if needed
                if st.session_state.highlight_text:
                    highlight_text = st.session_state.highlight_text
                    escaped_text = re.escape(highlight_text)
                    highlighted_content = re.sub(
                        f"({escaped_text})", 
                        r"<mark style='background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;'>\1</mark>", 
                        content, 
                        flags=re.IGNORECASE
                    )
                    st.markdown(highlighted_content, unsafe_allow_html=True)
                else:
                    st.markdown(content)
        else:
            st.info("Click on a reference button from the chat to view document content here.")
            
            # Show example of expected AWS response format
            with st.expander("ℹ️ AWS Knowledge Bases Integration Info"):
                st.write("This interface is designed to work with AWS Knowledge Bases output format:")
                st.json({
                    "citations": [
                        {
                            "generatedResponsePart": {
                                "textResponsePart": {
                                    "span": {"end": 123, "start": 0},
                                    "text": "Generated response text"
                                }
                            },
                            "retrievedReferences": [
                                {
                                    "content": {"text": "Reference text", "type": "TEXT"},
                                    "location": {
                                        "s3Location": {"uri": "s3://bucket/document.pdf"},
                                        "type": "S3"
                                    }
                                }
                            ]
                        }
                    ],
                    "output": {
                        "text": "Complete generated response with references [1]"
                    }
                })
    
    def handle_user_input(self):
        """Handle user input and generate responses"""
        user_input = st.chat_input("Ask me anything about your documents...")
        
        if user_input:
            # Add user message
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "type": "user",
                "content": user_input,
                "timestamp": timestamp
            })
            
            try:
                # Simulate AWS Knowledge Bases call
                aws_response = self.simulate_aws_knowledge_base_call(user_input)
                
                # Parse the AWS response
                parsed_response = self.parser.parse_aws_response(aws_response)
                
                if parsed_response["success"]:
                    # Add assistant response
                    st.session_state.messages.append({
                        "type": "assistant",
                        "content": parsed_response["response"],
                        "references": parsed_response["references"],
                        "timestamp": timestamp
                    })
                else:
                    # Handle parsing error
                    st.session_state.messages.append({
                        "type": "assistant",
                        "content": f"Sorry, I encountered an error: {parsed_response.get('error', 'Unknown error')}",
                        "references": [],
                        "timestamp": timestamp
                    })
                    
            except Exception as e:
                st.error(f"Error processing request: {str(e)}")
            
            st.rerun()
    
    def run(self):
        """Run the complete Streamlit application"""
        st.title("🤖 AWS Knowledge Bases RAG Chat Interface")
        st.markdown("Ask questions about your documents indexed in AWS Knowledge Bases")
        
        # Render main interface
        self.render_chat_interface()
        
        # Handle user input
        self.handle_user_input()

# Main application entry point
def main():
    """Main application function"""
    interface = StreamlitRAGInterface()
    interface.run()

# Integration instructions and example usage
def integrate_with_aws_knowledge_bases():
    """
    Example of how to integrate with actual AWS Knowledge Bases
    
    Replace the simulate_aws_knowledge_base_call method with:
    """
    import boto3
    
    # Initialize AWS client
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
    
    def query_knowledge_base(query: str, knowledge_base_id: str, model_arn: str) -> Dict[str, Any]:
        """
        Query AWS Knowledge Bases
        
        Args:
            query: User's question
            knowledge_base_id: Your Knowledge Base ID
            model_arn: ARN of the foundation model to use
            
        Returns:
            AWS Knowledge Bases response
        """
        try:
            response = bedrock_agent_runtime.retrieve_and_generate(
                input={
                    'text': query
                },
                retrieveAndGenerateConfiguration={
                    'type': 'KNOWLEDGE_BASE',
                    'knowledgeBaseConfiguration': {
                        'knowledgeBaseId': knowledge_base_id,
                        'modelArn': model_arn
                    }
                }
            )
            return response
        except Exception as e:
            print(f"Error querying Knowledge Base: {e}")
            return None

if __name__ == "__main__":
    main()