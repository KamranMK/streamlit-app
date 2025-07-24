import streamlit as st
import requests
import random
import re
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Chat with Markdown Viewer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "markdown_content" not in st.session_state:
    st.session_state.markdown_content = ""
if "current_file" not in st.session_state:
    st.session_state.current_file = ""
if "show_markdown_viewer" not in st.session_state:
    st.session_state.show_markdown_viewer = False

# Dummy responses with multiple S3 references embedded in text
DUMMY_RESPONSES = [
    {
        "text": """## Market Analysis Results 

Based on our comprehensive research, the current market shows significant growth potential in several key areas. The **renewable energy sector** is experiencing unprecedented expansion [^1], with solar installations increasing by 40% year-over-year [^2].

Key findings from our analysis:

- **Market Size**: The global market is projected to reach $2.8 trillion by 2025 [^3]
- **Consumer Behavior**: Recent surveys indicate a 65% preference shift toward sustainable products [^4]
- **Investment Trends**: Venture capital funding has increased by 180% in clean tech startups [^1]

For detailed implementation strategies, refer to our comprehensive guide [^5]. The regulatory landscape analysis shows favorable conditions across most markets [^2].

**Risk Assessment**: While opportunities are abundant, potential challenges include supply chain disruptions and regulatory changes [^3].
        """,
        "references": {
            "1": {
                "title": "Renewable Energy Market Report 2025",
                "url": "https://your-bucket.s3.amazonaws.com/research/renewable-energy-2025.md"
            },
            "2": {
                "title": "Solar Installation Growth Analysis",
                "url": "https://your-bucket.s3.amazonaws.com/research/solar-growth-analysis.md"
            },
            "3": {
                "title": "Global Market Projections",
                "url": "https://your-bucket.s3.amazonaws.com/research/market-projections.md"  
            },
            "4": {
                "title": "Consumer Behavior Survey Results",  
                "url": "https://your-bucket.s3.amazonaws.com/research/consumer-behavior.md"
            },
            "5": {
                "title": "Implementation Strategy Guide",
                "url": "https://your-bucket.s3.amazonaws.com/guides/implementation-strategy.md"
            }
        }
    },
    {
        "text": """## Technology Stack Recommendations

After analyzing current industry trends and performance benchmarks, here are our key recommendations for your technology stack:

**Frontend Development**: React remains the dominant choice [^1], with Next.js showing strong adoption for full-stack applications [^2]. TypeScript usage has grown to 78% among enterprise projects [^1].

**Backend Architecture**: 
- Microservices architecture is preferred for scalable applications [^3]
- Node.js and Python continue to lead in backend development [^4]
- GraphQL adoption is accelerating, especially for complex data requirements [^2]

**Database Solutions**: PostgreSQL maintains its position as the most trusted relational database [^5], while MongoDB leads in document storage solutions [^3].

**DevOps & Deployment**: Kubernetes orchestration is now standard for enterprise deployments [^4], with Docker containerization being nearly universal [^5].

For detailed setup instructions and best practices, consult our complete technical guide [^1].
        """,
        "references": {
            "1": {
                "title": "Frontend Development Trends 2025",
                "url": "https://your-bucket.s3.amazonaws.com/tech/frontend-trends-2025.md"
            },
            "2": {
                "title": "Full-Stack Framework Analysis", 
                "url": "https://your-bucket.s3.amazonaws.com/tech/fullstack-frameworks.md"
            },
            "3": {
                "title": "Microservices Architecture Guide",
                "url": "https://your-bucket.s3.amazonaws.com/architecture/microservices-guide.md"
            },
            "4": {
                "title": "Backend Development Survey",
                "url": "https://your-bucket.s3.amazonaws.com/tech/backend-survey.md"
            },
            "5": {
                "title": "Database Performance Benchmarks",
                "url": "https://your-bucket.s3.amazonaws.com/data/database-benchmarks.md"
            }
        }
    },
    {
        "text": """## User Experience Research Insights

Our recent UX research across 2,500 participants reveals critical insights for product development:

**Navigation Patterns**: Users expect intuitive navigation with no more than 3 clicks to reach any content [^1]. Mobile-first design is no longer optional, with 72% of users primarily accessing applications via mobile devices [^2].

**Performance Expectations**: 
- Page load times exceeding 3 seconds result in 53% user abandonment [^3]
- Interactive elements must respond within 100ms to feel instantaneous [^1]
- Users prefer progressive loading over traditional loading screens [^4]

**Accessibility Standards**: WCAG 2.1 AA compliance is now baseline expectation [^5], with screen reader compatibility being critical for enterprise adoption [^2].

**Content Strategy**: Scannable content with clear hierarchies performs 40% better in user testing [^3]. Visual elements should support, not replace, textual information [^4].

Implementation of these findings can significantly improve user engagement and retention rates [^5].
        """,
        "references": {
            "1": {
                "title": "Navigation UX Best Practices",
                "url": "https://your-bucket.s3.amazonaws.com/ux/navigation-best-practices.md"
            },
            "2": {
                "title": "Mobile-First Design Research",
                "url": "https://your-bucket.s3.amazonaws.com/ux/mobile-first-research.md"
            },
            "3": {
                "title": "Performance Impact on User Behavior",
                "url": "https://your-bucket.s3.amazonaws.com/performance/user-behavior-study.md"
            },
            "4": {
                "title": "Progressive Loading Techniques",
                "url": "https://your-bucket.s3.amazonaws.com/performance/progressive-loading.md"
            },
            "5": {
                "title": "Accessibility Implementation Guide",
                "url": "https://your-bucket.s3.amazonaws.com/accessibility/implementation-guide.md"
            }
        }
    }
]

def load_markdown_from_s3(url):
    """Load markdown content from S3 URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"# Error Loading Content\n\nFailed to load markdown file from: {url}\n\nError: {str(e)}\n\n*Note: This is a demo app. Please replace with actual S3 URLs containing markdown files.*"

def generate_dummy_response():
    """Generate a dummy chat response with multiple embedded references"""
    response_data = random.choice(DUMMY_RESPONSES)
    return response_data

def render_text_with_references(text, references, message_id):
    """Render text with clickable reference links"""
    # Display the main text as markdown
    st.markdown(text)
    
    # Add references section
    if references:
        st.markdown("---")
        st.markdown("**📚 References:**")
        
        # Create columns for reference buttons (2 per row)
        ref_items = list(references.items())
        for i in range(0, len(ref_items), 2):
            cols = st.columns(2)
            for j, (ref_num, ref_data) in enumerate(ref_items[i:i+2]):
                with cols[j]:
                    button_key = f"ref_{message_id}_{ref_num}"
                    if st.button(f"[{ref_num}] {ref_data['title']}", key=button_key, use_container_width=True):
                        with st.spinner(f"Loading {ref_data['title']}..."):
                            content = load_markdown_from_s3(ref_data['url'])
                            st.session_state.markdown_content = content
                            st.session_state.current_file = ref_data['url']
                            st.session_state.show_markdown_viewer = True
                        st.rerun()

# Custom CSS for better styling
st.markdown("""
<style>
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
}
.user-message {
    background-color: #e3f2fd;
    align-self: flex-end;
}
.assistant-message {
    background-color: #f5f5f5;
    align-self: flex-start;
}
.markdown-viewer {
    border: 1px solid #ddd;
    border-radius: 0.5rem;
    padding: 1rem;
    background-color: #fafafa;
    max-height: 80vh;
    overflow-y: auto;
}
.stButton > button {
    background-color: #1976d2;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
    cursor: pointer;
    margin-top: 0.5rem;
    font-size: 0.85rem;
}
.stButton > button:hover {
    background-color: #1565c0;
}
/* Reference button styling */
div[data-testid="column"] .stButton > button {
    background-color: #f8f9fa;
    color: #495057;
    border: 1px solid #dee2e6;
    padding: 0.4rem 0.8rem;
    font-size: 0.8rem;
    text-align: left;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
div[data-testid="column"] .stButton > button:hover {
    background-color: #e9ecef;
    border-color: #adb5bd;
}
</style>
""", unsafe_allow_html=True)

# Main layout - conditionally show columns based on markdown viewer state
if st.session_state.show_markdown_viewer:
    col1, col2 = st.columns([1, 1])
else:
    col1 = st.container()
    col2 = None

# Chat section
with col1:
    st.header("💬 Chat")
    
    # Chat messages container with scrolling
    if st.session_state.messages:
        # Display chat messages without the problematic HTML container
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"**{message['timestamp']}** - {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{message['timestamp']}**")
                    # Render message with embedded references
                    render_text_with_references(
                        message["content"], 
                        message.get("references", {}), 
                        i
                    )
    else:
        st.info("👋 Welcome! Start a conversation by typing a message below.")
    
    # Fixed chat input at bottom
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        # Generate dummy response
        response_data = generate_dummy_response()
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data["text"],
            "references": response_data["references"],
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.rerun()

# Markdown viewer section - only show if a link has been clicked
if st.session_state.show_markdown_viewer and col2 is not None:
    with col2:
        st.header("📋 Markdown Viewer")
        
        if st.session_state.current_file:
            st.caption(f"**Current file:** {st.session_state.current_file}")
            
            # Close viewer button
            if st.button("❌ Close Viewer"):
                st.session_state.show_markdown_viewer = False
                st.session_state.markdown_content = ""
                st.session_state.current_file = ""
                st.rerun()
        
        # Markdown viewer container
        viewer_container = st.container()
        with viewer_container:
            if st.session_state.markdown_content:
                st.markdown(
                    f'<div class="markdown-viewer">{st.session_state.markdown_content}</div>',
                    unsafe_allow_html=True
                )
                # Also render the markdown content properly
                st.markdown("---")
                st.markdown("### Rendered Content:")
                st.markdown(st.session_state.markdown_content)

# Footer
st.markdown("---")
st.caption("💡 **Demo App**: This app shows dummy responses with S3 markdown links. Replace the S3 URLs in the code with your actual markdown files.")