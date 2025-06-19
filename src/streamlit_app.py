import streamlit as st
import base64
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from main_v6 import app, RAG_AVAILABLE
from langchain_core.messages import HumanMessage

# Define emoji icons for each role
AVATAR_ICONS = {
    "user": "👤",
    "assistant": "🤖",
    "ceo": "💼",
    "cto": "💻",
    "cfo": "💰",
    "coo": "⚙️",
    "final_report": "📋"
}

def clean_text_for_pdf(text):
    """Clean text to remove Unicode characters that can't be encoded in latin-1"""
    replacements = {
        '\u2022': '•', '\u2013': '-', '\u2014': '--', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u00a0': ' '
    }
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        cleaned_text = ""
        for char in text:
            try:
                char.encode('latin-1')
                cleaned_text += char
            except UnicodeEncodeError:
                cleaned_text += '?'
        return cleaned_text

def create_pdf_download_link(content, filename):
    """Create PDF from text content with proper Unicode handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 16)
        
        # Add title with RAG indicator
        title = 'RAG-ENHANCED STARTUP CONSULTATION REPORT' if RAG_AVAILABLE else 'STARTUP CONSULTATION REPORT'
        pdf.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        clean_content = clean_text_for_pdf(content)
        pdf.set_font('Helvetica', '', 11)
        lines = clean_content.split('\n')
        
        for line in lines:
            if line.strip():
                if line.startswith('##'):
                    pdf.set_font('Helvetica', 'B', 14)
                    clean_line = line.replace('##', '').strip()
                    pdf.cell(0, 8, clean_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.ln(2)
                    pdf.set_font('Helvetica', '', 11)
                elif line.startswith('**') and line.endswith('**'):
                    pdf.set_font('Helvetica', 'B', 11)
                    clean_line = line.replace('**', '')
                    pdf.cell(0, 6, clean_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font('Helvetica', '', 11)
                elif line.startswith('• '):
                    bullet_line = line
                    if len(bullet_line) > 85:
                        words = bullet_line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 85:
                                current_line += word + " "
                            else:
                                if current_line.strip():
                                    pdf.cell(0, 5, current_line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                                current_line = "  " + word + " "
                        if current_line.strip():
                            pdf.cell(0, 5, current_line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    else:
                        pdf.cell(0, 5, bullet_line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                else:
                    if len(line) > 90:
                        words = line.split(' ')
                        current_line = ""
                        for word in words:
                            if len(current_line + word) < 90:
                                current_line += word + " "
                            else:
                                if current_line.strip():
                                    pdf.cell(0, 5, current_line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                                current_line = word + " "
                        if current_line.strip():
                            pdf.cell(0, 5, current_line.strip(), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    else:
                        pdf.cell(0, 5, line, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.ln(3)
        
        pdf_output = pdf.output(dest="S")
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
        
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📄 Download PDF Report</a>'
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def extract_final_report(messages):
    """Enhanced final report detection"""
    for message in reversed(messages):
        content = message.get("content", "")
        if any(indicator in content for indicator in [
            "FINAL REPORT:", "FINAL REPORT ", "Final Report:", "Executive Summary"
        ]) and len(content) > 200:
            return content
    
    if len(messages) >= 8:
        ceo_messages = [msg for msg in messages if msg.get("role") == "ceo"]
        if ceo_messages:
            longest_ceo_message = max(ceo_messages, key=lambda x: len(x.get("content", "")))
            if len(longest_ceo_message.get("content", "")) > 500:
                return longest_ceo_message["content"]
    return None

# Set page config
st.set_page_config(page_title="RAG-Enhanced AI Startup Consultancy", layout="wide")

# Dynamic title based on RAG availability
if RAG_AVAILABLE:
    st.title("🚀 RAG-Enhanced AI Startup Consultancy Firm")
    st.markdown("*Powered by Knowledge Bases + Real-time Search*")
else:
    st.title("🚀 AI Startup Consultancy Firm")
    st.markdown("*Powered by Real-time Search*")

# Sidebar
with st.sidebar:
    st.header("About This App")
    if RAG_AVAILABLE:
        st.write("Our AI board (CEO, CTO, CFO, COO) will analyze your business idea using RAG-enhanced knowledge bases and real-time market data.")
    else:
        st.write("Our AI board (CEO, CTO, CFO, COO) will analyze your business idea using real-time market data.")
    
    st.markdown("---")
    st.subheader("How It Works")
    st.write("1. **Submit Idea:** Enter your business concept")
    st.write("2. **Board Discussion:** Watch AI agents collaborate")
    if RAG_AVAILABLE:
        st.write("3. **RAG Enhancement:** Agents access knowledge bases")
        st.write("4. **Final Report:** Get comprehensive analysis")
        st.write("5. **Export:** Download as PDF or Markdown")
    else:
        st.write("3. **Final Report:** Get comprehensive analysis")
        st.write("4. **Export:** Download as PDF or Markdown")
    
    st.markdown("---")
    
    # System status
    st.subheader("System Status")
    if RAG_AVAILABLE:
        st.success("🔍 RAG System: Active")
        st.info("📚 Knowledge Bases: Loaded")
    else:
        st.warning("🔍 RAG System: Not Available")
        st.info("🌐 TavilySearch: Active")
    
    st.markdown("---")
    st.caption("Powered by LangGraph • OpenAI • Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    welcome_message = """👋 Welcome to your AI Startup Consultancy! I'm here to facilitate a board meeting with our expert AI agents.

**Ready to analyze your business idea?** Our team includes:
- 💼 **Sarah (CEO)**: Strategic vision and market analysis
- 💻 **Mike (CTO)**: Technical feasibility and architecture  
- 💰 **Jennifer (CFO)**: Financial modeling and funding strategy
- ⚙️ **Tom (COO)**: Operations and execution planning
"""
    
    if RAG_AVAILABLE:
        welcome_message += "\n🔍 **Enhanced with RAG**: Agents now have access to specialized knowledge bases for more accurate insights!"
    
    welcome_message += "\nPlease share your business idea below to begin the consultation!"
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": welcome_message
    })

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        display_name = "You"
        avatar = AVATAR_ICONS["user"]
    elif role == "assistant":
        display_name = "Assistant"
        avatar = AVATAR_ICONS["assistant"]
    elif role in ["ceo", "cto", "cfo", "coo"]:
        names = {"ceo": "Sarah (CEO)", "cto": "Mike (CTO)", "cfo": "Jennifer (CFO)", "coo": "Tom (COO)"}
        display_name = names[role]
        avatar = AVATAR_ICONS[role]
    else:
        display_name = role.upper()
        avatar = "💬"
    
    with st.chat_message(name=display_name, avatar=avatar):
        st.markdown(content)

# Handle user input
if prompt := st.chat_input("Describe your business idea..."):
    if len(prompt.strip()) < 10:
        st.warning("Please provide a more detailed business idea (at least 10 characters)")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("You", avatar=AVATAR_ICONS["user"]):
        st.markdown(prompt)
    
    # Prepare initial message
    initial_messages = [HumanMessage(content=f"Analyze this business idea and provide comprehensive consultation: {prompt}")]
    
    # Prepare initial state for RAG-enhanced system
    initial_state = {
        "messages": initial_messages,
        "discussion_phase": "initial",
        "topics_discussed": [],
        "pending_questions": [],
        "message_count": 0,
        "last_speaker": "",
        "agent_participation": {"CEO": False, "CTO": False, "CFO": False, "COO": False},
        "response_hashes": {},
        "agent_call_counts": {"CEO": 0, "CTO": 0, "CFO": 0, "COO": 0},
        "conversation_quality": 1.0,
        "context_summary": ""
    }
    
    # Process with spinner
    spinner_text = "🤝 The RAG-enhanced AI board is in session..." if RAG_AVAILABLE else "🤝 The AI board is in session..."
    with st.spinner(spinner_text):
        try:
            config = {"recursion_limit": 30}
            
            for output in app.stream(initial_state, config=config):
                for key, value in output.items():
                    if key != "__end__" and "messages" in value and value["messages"]:
                        agent_role = key.lower()
                        agent_message = value['messages'][-1].content
                        
                        # Determine display info
                        names = {"ceo": "Sarah (CEO)", "cto": "Mike (CTO)", "cfo": "Jennifer (CFO)", "coo": "Tom (COO)"}
                        display_name = names.get(agent_role, key.upper())
                        avatar = AVATAR_ICONS.get(agent_role, "💬")
                        
                        # Show quality indicator if available
                        quality = value.get('conversation_quality', 1.0)
                        quality_indicator = "🟢" if quality > 0.7 else "🟡" if quality > 0.4 else "🔴"
                        
                        # Display message
                        with st.chat_message(f"{display_name} {quality_indicator}", avatar=avatar):
                            st.markdown(agent_message)
                        
                        # Add to session state
                        st.session_state.messages.append({
                            "role": agent_role,
                            "content": agent_message
                        })
                        
        except Exception as e:
            st.error(f"An error occurred during the consultation: {str(e)}")
            st.info("Please try again or rephrase your business idea.")

# Export functionality
if len(st.session_state.messages) > 2:
    final_report = extract_final_report(st.session_state.messages)
    if final_report:
        st.markdown("---")
        
        # Dynamic export title
        if RAG_AVAILABLE:
            st.subheader("📄 Export Your RAG-Enhanced Consultation Report")
        else:
            st.subheader("📄 Export Your Consultation Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    filename = "rag_enhanced_startup_report.pdf" if RAG_AVAILABLE else "startup_consultation_report.pdf"
                    pdf_link = create_pdf_download_link(final_report, filename)
                    if pdf_link:
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        st.success("PDF download link generated!")
                    else:
                        st.error("Failed to generate PDF. Try downloading as Markdown instead.")
        
        with col2:
            clean_markdown = clean_text_for_pdf(final_report)
            filename = "rag_enhanced_startup_report.md" if RAG_AVAILABLE else "startup_consultation_report.md"
            st.download_button(
                label="📝 Download Markdown",
                data=clean_markdown,
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 New Consultation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
                
    elif len(st.session_state.messages) > 15:
        st.markdown("---")
        st.info("💡 The consultation is in progress. Export options will appear once the final report is ready.")
        if st.button("🔄 Start New Consultation"):
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
if RAG_AVAILABLE:
    st.markdown(
        '<div style="text-align: center; color: #888;">🔍 RAG-Enhanced AI Startup Consultancy • Knowledge-Powered Business Intelligence</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align: center; color: #888;">🔍 AI Startup Consultancy • Real-time Market Intelligence</div>',
        unsafe_allow_html=True
    )

