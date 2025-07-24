import streamlit as st
import random
import re
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Fixed Layout Chat",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "markdown_content" not in st.session_state:
    st.session_state.markdown_content = ""
if "current_file" not in st.session_state:
    st.session_state.current_file = ""
if "show_markdown_viewer" not in st.session_state:
    st.session_state.show_markdown_viewer = False

# --- DUMMY DATA AND MOCKS (Copied from original) ---
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
            "1": {"title": "Renewable Energy Market Report 2025", "url": "https://your-bucket.s3.amazonaws.com/research/renewable-energy-2025.md"},
            "2": {"title": "Solar Installation Growth Analysis", "url": "https://your-bucket.s3.amazonaws.com/research/solar-growth-analysis.md"},
            "3": {"title": "Global Market Projections", "url": "https://your-bucket.s3.amazonaws.com/research/market-projections.md"},
            "4": {"title": "Consumer Behavior Survey Results", "url": "https://your-bucket.s3.amazonaws.com/research/consumer-behavior.md"},
            "5": {"title": "Implementation Strategy Guide", "url": "https://your-bucket.s3.amazonaws.com/guides/implementation-strategy.md"}
        }
    },
]
MOCK_S3_FILES = {
    "https://your-bucket.s3.amazonaws.com/research/renewable-energy-2025.md": "# Renewable Energy Market Report 2025\n\n...",
    "https://your-bucket.s3.amazonaws.com/research/solar-growth-analysis.md": "# Solar Installation Growth Analysis\n\n...",
    "https://your-bucket.s3.amazonaws.com/research/market-projections.md": "# Global Market Projections\n\n...",
    "https://your-bucket.s3.amazonaws.com/research/consumer-behavior.md": "# Consumer Behavior Survey Results\n\n...",
    "https://your-bucket.s3.amazonaws.com/guides/implementation-strategy.md": "# Implementation Strategy Guide\n\n...",
}
# Note: Full dummy data is omitted for brevity but should be included from the original script.

# --- HELPER FUNCTIONS (Slightly modified to allow HTML) ---
def load_markdown_from_s3(url):
    """Load markdown content from S3 URL (mocked for demo)"""
    return MOCK_S3_FILES.get(url, f"# Mock File\n\nContent for {url}")

def highlight_matching_text(markdown_content, reference_text_snippets):
    """Highlight text in markdown that matches reference snippets"""
    if not reference_text_snippets:
        return markdown_content
    highlighted_content = markdown_content
    for snippet in reference_text_snippets:
        clean_snippet = re.sub(r'\*\*([^*]+)\*\*', r'\1', snippet)
        clean_snippet = re.sub(r'[^\w\s%]', '', clean_snippet)
        words = clean_snippet.split()
        if len(words) >= 2:
            for i in range(len(words) - 1):
                for length in range(4, 1, -1):
                    if i + length <= len(words):
                        phrase = ' '.join(words[i:i+length])
                        if len(phrase) > 6:
                            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                            highlighted_content = pattern.sub(
                                f'<mark style="background-color: #ffeb3b; padding: 2px; border-radius: 3px;">{phrase}</mark>',
                                highlighted_content, count=3
                            )
    return highlighted_content

def generate_dummy_response():
    """Generate a dummy chat response"""
    return random.choice(DUMMY_RESPONSES)

# --- CSS FOR FIXED LAYOUT ---
st.markdown("""
<style>
    /* Reset Streamlit's default padding and margins */
    .main .block-container {
        padding: 0px !important;
        max-width: 100% !important;
    }
    
    /* Hide Streamlit's default header and toolbar */
    header[data-testid="stHeader"], .stToolbar {
        display: none !important;
    }
    
    /* Main container for the entire app layout */
    .app-container {
        display: flex;
        flex-direction: row;
        height: 100vh;
        width: 100%;
        overflow: hidden;
    }

    /* Base style for both columns */
    .column {
        height: 100%;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
    }
    
    .chat-column {
        padding: 1rem;
    }

    .viewer-column {
        padding: 1rem;
        border-left: 1px solid #e0e0e0;
    }
    
    .column-header {
        flex-shrink: 0;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    
    /* Scrollable content area */
    .scrollable-content {
        flex-grow: 1;
        overflow-y: auto;
        min-height: 0;
        padding-right: 1rem; /* Space for scrollbar */
    }

    /* Fixed chat input area */
    .input-container {
        flex-shrink: 0;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }

    /* Custom scrollbar */
    .scrollable-content::-webkit-scrollbar { width: 8px; }
    .scrollable-content::-webkit-scrollbar-track { background: transparent; }
    .scrollable-content::-webkit-scrollbar-thumb {
        background-color: #ccc;
        border-radius: 10px;
        border: 2px solid #f1f1f1;
    }

    /* Viewer header layout */
    .viewer-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>
""", unsafe_allow_html=True)

def render_text_with_references(text, references, message_id):
    """Render text with clickable reference links"""
    st.markdown(text, unsafe_allow_html=True) # Allow HTML for highlights
    if references:
        st.markdown("---")
        st.markdown("**📚 References:**")
        ref_items = list(references.items())
        for i in range(0, len(ref_items), 2):
            cols = st.columns(2)
            for j, (ref_num, ref_data) in enumerate(ref_items[i:i+2]):
                with cols[j]:
                    button_key = f"ref_{message_id}_{ref_num}"
                    if st.button(f"[{ref_num}] {ref_data['title']}", key=button_key, use_container_width=True):
                        content = load_markdown_from_s3(ref_data['url'])
                        ref_pattern = rf'\[\^{ref_num}\]'
                        matches = list(re.finditer(ref_pattern, text))
                        snippets = [text[max(0, m.start()-100):min(len(text), m.end()+100)] for m in matches]
                        st.session_state.markdown_content = highlight_matching_text(content, snippets)
                        st.session_state.current_file = ref_data['url']
                        st.session_state.show_markdown_viewer = True
                        st.rerun()

# --- APP LAYOUT ---
# Define the flex ratio for the columns
chat_flex = "1" if st.session_state.show_markdown_viewer else "2"
viewer_flex = "1"

st.markdown('<div class="app-container">', unsafe_allow_html=True)

# --- CHAT COLUMN (Left Side) ---
st.markdown(f'<div class="column chat-column" style="flex: {chat_flex};">', unsafe_allow_html=True)

st.markdown('<div class="column-header"><h3>💬 Chat</h3></div>', unsafe_allow_html=True)

# Scrollable Message Area
with st.container():
    st.markdown('<div id="message-container" class="scrollable-content">', unsafe_allow_html=True)
    if st.session_state.messages:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.write(f"**{msg['timestamp']}**")
                if msg["role"] == "user":
                    st.write(msg['content'])
                else:
                    render_text_with_references(msg["content"], msg.get("references", {}), i)
    else:
        st.info("👋 Welcome! Start the conversation below.")
    st.markdown('</div>', unsafe_allow_html=True)

# Fixed Input Area
st.markdown('<div class="input-container">', unsafe_allow_html=True)
if prompt := st.chat_input("Type your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M")})
    response_data = generate_dummy_response()
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response_data["text"], 
        "references": response_data["references"], 
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True) # Close chat-column

# --- MARKDOWN VIEWER COLUMN (Right Side) ---
if st.session_state.show_markdown_viewer:
    st.markdown(f'<div class="column viewer-column" style="flex: {viewer_flex};">', unsafe_allow_html=True)
    
    st.markdown('<div class="column-header viewer-header">', unsafe_allow_html=True)
    st.markdown("<h3>📋 Document Viewer</h3>", unsafe_allow_html=True)
    if st.button("❌ Close", key="close_viewer"):
        st.session_state.show_markdown_viewer = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="scrollable-content">', unsafe_allow_html=True)
    if st.session_state.current_file:
        filename = st.session_state.current_file.split('/')[-1]
        st.markdown(f"**File:** `{filename}`")
        st.markdown("---")
    st.markdown(st.session_state.markdown_content, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close app-container

# --- JAVASCRIPT FOR AUTO-SCROLL ---
# This script runs on every rerun to scroll the message container down.
st.components.v1.html("""
<script>
    const messageContainer = document.getElementById('message-container');
    if (messageContainer) {
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
</script>
""", height=0)