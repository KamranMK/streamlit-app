import streamlit as st
import requests
import random
import re
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Chat with Markdown Viewer",
    layout="wide",
    initial_sidebar_state="expanded",
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

# Mock S3 markdown files for demo purposes
MOCK_S3_FILES = {
    "https://your-bucket.s3.amazonaws.com/research/renewable-energy-2025.md": """# Renewable Energy Market Report 2025

## Executive Summary
The renewable energy sector is experiencing **unprecedented expansion** with record-breaking growth across all major technologies.

## Key Findings
- **Solar installations** increased by **40% year-over-year**
- Wind capacity additions reached 95 GW globally
- **Investment trends** show **180% increase** in clean tech startup funding
- Grid storage deployments doubled compared to 2024

## Market Dynamics
The market transformation is driven by:
1. Declining technology costs
2. Supportive government policies  
3. Corporate sustainability commitments
4. Investor preference for ESG assets

## Regional Analysis
- **Asia Pacific**: Leading in solar installations
- **Europe**: Offshore wind expansion
- **North America**: Storage and grid modernization

## Technology Outlook
Next-generation technologies showing promise:
- **Perovskite solar cells** - 31% efficiency achieved
- **Floating wind turbines** - Accessing deeper waters
- **Green hydrogen** - Production costs declining rapidly

*Data compiled from 200+ industry sources and expert interviews.*
""",

    "https://your-bucket.s3.amazonaws.com/research/solar-growth-analysis.md": """# Solar Installation Growth Analysis

## Growth Metrics 2025
The solar industry has achieved remarkable milestones in 2025:

### Installation Statistics
- **Residential solar**: 28% growth rate
- **Commercial installations**: 35% increase
- **Utility-scale projects**: **40% year-over-year growth**
- Total capacity additions: 191 GW globally

### Cost Reductions
- **Module prices**: Down 15% from 2024
- **Installation costs**: Reduced by 12%
- **Financing rates**: Improved accessibility

### Market Drivers
Key factors accelerating adoption:
1. **Regulatory landscape** shows **favorable conditions**
2. Net metering policies expansion
3. Corporate renewable energy procurement
4. Grid parity achievement in 140+ markets

### Regional Performance
- **United States**: 32 GW of new capacity
- **China**: Maintained manufacturing leadership
- **India**: Surpassed 75 GW total installed capacity
- **Europe**: 42 GW additions despite supply challenges

### Technology Advances
- **Bifacial panels**: 65% market share
- **Tracking systems**: 85% of utility installations
- **Energy storage integration**: 40% co-location rate

*Analysis based on installation data from 50+ countries.*
""",

    "https://your-bucket.s3.amazonaws.com/research/market-projections.md": """# Global Market Projections

## Market Size Forecast
The **global market is projected to reach $2.8 trillion by 2025**, representing unprecedented growth in the clean energy sector.

## Sector Breakdown
### Energy Technologies ($1.8T)
- Solar photovoltaics: $580B
- Wind power: $420B  
- Energy storage: $340B
- Grid infrastructure: $460B

### Transportation ($650B)
- Electric vehicles: $480B
- Charging infrastructure: $95B
- Alternative fuels: $75B

### Industrial Applications ($350B)
- Green hydrogen: $120B
- Carbon capture: $90B
- Industrial heat pumps: $85B
- Process optimization: $55B

## Growth Drivers
1. **Policy Support**: $4.2T in announced government commitments
2. **Corporate Demand**: 400+ companies with net-zero targets
3. **Technology Maturity**: Cost-competitive solutions
4. **Capital Availability**: $1.8T in committed private investment

## Risk Factors
**Potential challenges** include:
- **Supply chain disruptions** and material constraints
- **Regulatory changes** affecting incentive structures
- Grid integration complexities
- Skilled workforce shortages

## Regional Outlook
- **Asia**: 45% of global market value
- **North America**: 28% market share
- **Europe**: 22% contribution
- **Rest of World**: 5% but fastest growing

*Projections based on analysis of 300+ market research reports.*
""",

    "https://your-bucket.s3.amazonaws.com/research/consumer-behavior.md": """# Consumer Behavior Survey Results

## Survey Overview
Comprehensive study of **2,500 participants** across 12 countries examining shifting preferences toward sustainable products.

## Key Findings

### Preference Shift
**65% preference shift toward sustainable products** represents the largest behavioral change in consumer markets since digital adoption.

### Demographics
- **Age 18-34**: 78% prioritize sustainability
- **Age 35-54**: 61% consider environmental impact
- **Age 55+**: 52% factor in sustainability

### Purchase Drivers
1. **Environmental impact**: 73% of respondents
2. **Long-term cost savings**: 68%
3. **Brand reputation**: 54%
4. **Product quality**: 49%
5. **Social influence**: 31%

### Willingness to Pay Premium
- **Solar panels**: 67% willing to pay 10-15% more
- **Electric vehicles**: 58% accept higher upfront costs
- **Energy-efficient appliances**: 71% prefer despite price
- **Green building materials**: 44% willing to invest

### Barriers to Adoption
- **High upfront costs**: 62% cite as primary concern
- **Limited product availability**: 45%
- **Uncertainty about performance**: 38%
- **Lack of information**: 33%

### Regional Variations
- **Scandinavia**: Highest adoption rates (81%)
- **Western Europe**: Strong sustainability focus (74%)
- **North America**: Growing awareness (66%)
- **Asia Pacific**: Rapid shift in urban areas (59%)

*Survey conducted by independent research firm with margin of error ±2.1%*
""",

    "https://your-bucket.s3.amazonaws.com/tech/frontend-trends-2025.md": """# Frontend Development Trends 2025

## Framework Landscape
**React remains the dominant choice** with 67% developer adoption, while new patterns emerge for modern web development.

### Framework Statistics
- **React**: 67% adoption rate
- **Vue.js**: 24% market share
- **Angular**: 18% enterprise usage
- **Svelte**: 12% growing rapidly
- **Solid.js**: 8% emerging contender

## TypeScript Adoption
**TypeScript usage has grown to 78% among enterprise projects**, marking a significant shift from JavaScript-only development.

### Benefits Driving Adoption
1. **Type safety**: Reduces runtime errors by 65%
2. **Developer experience**: Enhanced IDE support
3. **Code maintainability**: Easier refactoring
4. **Team collaboration**: Self-documenting code

## Modern Development Patterns
### Component Architecture
- **Composition over inheritance**: 89% prefer functional components
- **Custom hooks**: 72% use for state logic
- **Context API**: 84% for state management
- **Server components**: 31% adoption in Next.js apps

### Performance Optimization
- **Code splitting**: Universal adoption (95%)
- **Lazy loading**: 87% implement for images/components
- **Bundle optimization**: Tree shaking standard practice
- **Runtime performance**: Focus on Core Web Vitals

## Tooling Evolution
### Build Tools
- **Vite**: 58% adoption for new projects
- **Webpack**: 71% still in production use
- **Turbopack**: 15% early adoption
- **Rollup**: 23% for library development

### Development Experience
- **Hot module replacement**: Expected feature (98%)
- **Fast refresh**: Critical for React development
- **Source maps**: Essential for debugging
- **Dev server performance**: Sub-second startup times

*Data collected from 15,000+ developer surveys and GitHub analysis.*
""",

    "https://your-bucket.s3.amazonaws.com/tech/fullstack-frameworks.md": """# Full-Stack Framework Analysis

## Next.js Leadership
**Next.js showing strong adoption for full-stack applications** with 72% of React developers choosing it for new projects.

### Next.js Features Driving Adoption
- **App Router**: 89% positive developer feedback
- **Server Components**: Reducing client bundle sizes by 40%
- **API Routes**: Simplified backend development
- **Edge Runtime**: Sub-100ms response times
- **Built-in optimization**: Automatic image and font optimization

## Framework Comparison

### Next.js (React-based)
- **Adoption**: 72% of React developers
- **Performance**: Excellent Core Web Vitals scores
- **DX**: Comprehensive tooling and documentation
- **Ecosystem**: Extensive third-party integration

### Nuxt.js (Vue-based)
- **Adoption**: 68% of Vue developers  
- **SSR/SSG**: Best-in-class hybrid rendering
- **Module system**: Rich plugin ecosystem
- **Performance**: Auto-optimization features

### SvelteKit (Svelte-based)
- **Bundle size**: Smallest runtime footprint
- **Performance**: Fastest initial page loads
- **DX**: Intuitive API design
- **Adoption**: Growing in startups (45% increase)

### Remix (React-based)
- **Web standards**: Focus on platform APIs
- **Data loading**: Optimized server-client coordination
- **Progressive enhancement**: Works without JavaScript
- **Adoption**: 23% among React teams

## Enterprise Considerations
### Scalability Factors
1. **Team size**: Larger teams prefer Next.js/Nuxt
2. **Performance requirements**: SvelteKit for speed-critical apps
3. **SEO needs**: All frameworks support SSR/SSG
4. **Migration path**: Next.js easiest from existing React apps

### Technology Selection Criteria
- **Developer experience**: Primary factor (87%)
- **Performance**: Critical for user-facing apps (78%)
- **Ecosystem support**: Long-term viability (71%)
- **Learning curve**: Team skill alignment (65%)

*Analysis based on 500+ production deployments and developer interviews.*
"""
}

def load_markdown_from_s3(url):
    """Load markdown content from S3 URL (mocked for demo)"""
    if url in MOCK_S3_FILES:
        return MOCK_S3_FILES[url]
    else:
        return f"# Mock S3 File\n\nThis would normally load content from: {url}\n\n*Note: This is a demo app with mocked S3 content.*"

def highlight_matching_text(markdown_content, reference_text_snippets):
    """Highlight text in markdown that matches reference snippets"""
    if not reference_text_snippets:
        return markdown_content
    
    highlighted_content = markdown_content
    
    # Common phrases to look for based on the reference text
    for snippet in reference_text_snippets:
        # Remove markdown formatting and clean the snippet
        clean_snippet = re.sub(r'\*\*([^*]+)\*\*', r'\1', snippet)  # Remove bold markdown
        clean_snippet = re.sub(r'[^\w\s%]', '', clean_snippet)  # Remove punctuation except %
        
        # Split into words and look for key phrases
        words = clean_snippet.split()
        if len(words) >= 2:
            # Look for 2-4 word phrases
            for i in range(len(words) - 1):
                phrase_2 = ' '.join(words[i:i+2])
                phrase_3 = ' '.join(words[i:i+3]) if i+2 < len(words) else ''
                phrase_4 = ' '.join(words[i:i+4]) if i+3 < len(words) else ''
                
                for phrase in [phrase_4, phrase_3, phrase_2]:
                    if phrase and len(phrase) > 6:  # Only highlight meaningful phrases
                        # Case-insensitive search and highlight
                        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                        highlighted_content = pattern.sub(
                            f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px; font-weight: 600;">{phrase}</mark>',
                            highlighted_content,
                            count=3  # Limit highlights to avoid over-highlighting
                        )
    
    return highlighted_content

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
                            # Load content from mock S3
                            content = load_markdown_from_s3(ref_data['url'])
                            
                            # Extract text snippets that might be in the reference
                            reference_snippets = []
                            
                            # Try to extract relevant phrases from the original text near this reference
                            ref_pattern = rf'\[?\^{ref_num}\]?'
                            matches = list(re.finditer(ref_pattern, text))
                            for match in matches:
                                # Get surrounding context (100 chars before and after)
                                start = max(0, match.start() - 100)
                                end = min(len(text), match.end() + 100)
                                context = text[start:end]
                                reference_snippets.append(context)
                            
                            # Highlight matching content
                            highlighted_content = highlight_matching_text(content, reference_snippets)
                            
                            # Update session state
                            st.session_state.markdown_content = highlighted_content
                            st.session_state.current_file = ref_data['url']
                            st.session_state.show_markdown_viewer = True
                            
                            # Force rerun to show the content
                            st.rerun()

# Custom CSS for fixed layout
st.markdown("""
<style>
/* Remove all default Streamlit margins and padding */
.main .block-container {
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
}

/* Remove Streamlit's default spacing */
.element-container {
    margin: 0 !important;
    padding: 0 !important;
}

.stMarkdown {
    margin-bottom: 0 !important;
}

/* Hide Streamlit header */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Remove top toolbar */
.stToolbar {
    display: none !important;
}

/* Main app container */
.stApp {
    padding-top: 0 !important;
}

/* Column containers */
.stColumn > div {
    padding: 0 !important;
}


.chat-title {
    padding: 1rem 0 0.5rem 0;
    margin: 0;
    border-bottom: 2px solid #e0e0e0;
}

.chat-messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 1rem 0;
    min-height: 0;
}

.chat-input-area {
    padding: 0.5rem 0 1rem 0;
    border-top: 1px solid #e0e0e0;
}

/* Markdown viewer styling */
.viewer-section {
}

.viewer-header {
    padding: 1rem;
    background: white;
    border-bottom: 2px solid #e0e0e0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
}

.viewer-content {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    min-height: 0;
}

/* Button styling */
.stButton > button {
    background-color: #1976d2;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 0.25rem;
    font-size: 0.85rem;
}

.stButton > button:hover {
    background-color: #1565c0;
}

/* Reference buttons */
div[data-testid="column"] .stButton > button {
    background-color: #f8f9fa !important;
    color: #495057 !important;
    border: 1px solid #dee2e6 !important;
    padding: 0.4rem 0.8rem !important;
    font-size: 0.8rem !important;
    text-align: left !important;
}

div[data-testid="column"] .stButton > button:hover {
    background-color: #e9ecef !important;
}

/* Scrollbars */
.chat-messages-area::-webkit-scrollbar,
.viewer-content::-webkit-scrollbar {
    width: 6px;
}

.chat-messages-area::-webkit-scrollbar-track,
.viewer-content::-webkit-scrollbar-track {
    background: #f1f1f1;
}

.chat-messages-area::-webkit-scrollbar-thumb,
.viewer-content::-webkit-scrollbar-thumb {
    background: #c1c1c1;
    border-radius: 3px;
}

/* Chat input container fix */
.stChatInputContainer {
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Main layout - conditionally show columns based on markdown viewer state
if st.session_state.show_markdown_viewer:
    col1, col2 = st.columns([1, 1])
else:
    col1 = st.container()
    col2 = None

# Chat section with fixed layout
with col1:
    st.markdown('<div>', unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="chat-title">', unsafe_allow_html=True)
    st.header("💬 Chat")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Messages area
    st.markdown('<div class="chat-messages-area">', unsafe_allow_html=True)
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(f"**{message['timestamp']}** - {message['content']}")
            else:
                with st.chat_message("assistant"):
                    st.write(f"**{message['timestamp']}**")
                    render_text_with_references(
                        message["content"], 
                        message.get("references", {}), 
                        i
                    )
    else:
        st.info("👋 Welcome! Start a conversation by typing a message below.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M")
        })
        
        response_data = generate_dummy_response()
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_data["text"],
            "references": response_data["references"],
            "timestamp": datetime.now().strftime("%H:%M")
        })
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right column for markdown viewer
if st.session_state.show_markdown_viewer and col2 is not None:
    with col2:
        st.markdown('<div class="viewer-section">', unsafe_allow_html=True)
        
        # Header with close button
        st.markdown('<div class="viewer-header">', unsafe_allow_html=True)
        header_col1, header_col2 = st.columns([3, 1])
        with header_col1:
            st.markdown("### 📋 Document Viewer")
        with header_col2:
            if st.button("❌ Close", key="close_viewer", help="Close document viewer"):
                st.session_state.show_markdown_viewer = False
                st.session_state.markdown_content = ""
                st.session_state.current_file = ""
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Content area
        st.markdown('<div class="viewer-content">', unsafe_allow_html=True)
        
        if st.session_state.current_file:
            filename = st.session_state.current_file.split('/')[-1].replace('.md', '').replace('-', ' ').title()
            st.markdown(f"""
            <div style="background: linear-gradient(90deg, #3f51b5, #5c6bc0); color: white; padding: 1rem; 
                        border-radius: 0.5rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.2rem;">📄</span>
                <div>
                    <div style="font-weight: 600; font-size: 1.1rem;">{filename}</div>
                    <div style="font-size: 0.85rem; opacity: 0.9;">Source Document</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.markdown_content:
            st.markdown("#### 📖 Content Preview")
            
            st.markdown(
                f'<div style="background: white; padding: 1.5rem; border: 2px solid #3f51b5; '
                f'border-radius: 0.5rem; box-shadow: 0 4px 6px rgba(63, 81, 181, 0.1);">'
                f'{st.session_state.markdown_content}'
                f'</div>',
                unsafe_allow_html=True
            )
            
            st.info("💡 **Highlighted content:** Text matching your research query is highlighted in yellow.")
        else:
            st.info("Click a reference button to load document content")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("💡 **Demo App**: This app shows dummy responses with S3 markdown links. Replace the S3 URLs in the code with your actual markdown files.")