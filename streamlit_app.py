import streamlit as st
import base64
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from main_v4 import app
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
    # Replace common Unicode characters with latin-1 equivalents
    replacements = {
        '\u2022': '•',  # Bullet point -> ASCII bullet
        '\u2013': '-',  # En dash -> hyphen
        '\u2014': '--', # Em dash -> double hyphen
        '\u2018': "'",  # Left single quote -> apostrophe
        '\u2019': "'",  # Right single quote -> apostrophe
        '\u201c': '"',  # Left double quote -> quote
        '\u201d': '"',  # Right double quote -> quote
        '\u2026': '...',# Ellipsis -> three dots
        '\u00a0': ' ',  # Non-breaking space -> regular space
    }
    
    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)
    
    # Replace markdown bullet points with ASCII
    text = re.sub(r'^-\s+', '• ', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s+', '• ', text, flags=re.MULTILINE)
    
    # Remove or replace any remaining non-latin-1 characters
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        # If encoding still fails, replace problematic characters
        cleaned_text = ""
        for char in text:
            try:
                char.encode('latin-1')
                cleaned_text += char
            except UnicodeEncodeError:
                cleaned_text += '?'  # Replace with question mark
        return cleaned_text

def create_pdf_download_link(content, filename):
    """Create PDF from text content with proper Unicode handling"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Use updated fpdf2 syntax
        pdf.set_font('Helvetica', 'B', 16)
        
        # Add title with updated syntax
        pdf.cell(0, 10, 'STARTUP CONSULTATION REPORT', new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        pdf.ln(10)
        
        # Clean content for PDF compatibility
        clean_content = clean_text_for_pdf(content)
        
        # Process content
        pdf.set_font('Helvetica', '', 11)
        lines = clean_content.split('\n')
        
        for line in lines:
            if line.strip():
                # Handle markdown headers
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
                    # Handle bullet points with proper encoding
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
                    # Handle regular text with wrapping
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
        
        # Create download link
        pdf_output = pdf.output(dest="S")
        if isinstance(pdf_output, str):
            pdf_bytes = pdf_output.encode('latin-1')
        else:
            pdf_bytes = pdf_output
            
        b64 = base64.b64encode(pdf_bytes).decode()
        return f'<a href="data:application/pdf;base64,{b64}" download="{filename}.pdf">📄 Download PDF Report</a>'
        
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}")
        return None

def extract_final_report(messages):
    """Enhanced final report detection"""
    for message in reversed(messages):
        content = message.get("content", "")
        # Look for multiple indicators of final report
        if any(indicator in content for indicator in [
            "FINAL REPORT:", 
            "FINAL REPORT ", 
            "Final Report:",
            "Executive Summary"
        ]) and len(content) > 200:  # Ensure it's substantial
            return content
    
    # Fallback: Look for CEO's longest message if conversation is substantial
    if len(messages) >= 8:
        ceo_messages = [msg for msg in messages if msg.get("role") == "ceo"]
        if ceo_messages:
            longest_ceo_message = max(ceo_messages, key=lambda x: len(x.get("content", "")))
            if len(longest_ceo_message.get("content", "")) > 500:
                return longest_ceo_message["content"]
    
    return None


# Set page config
st.set_page_config(page_title="AI Startup Consultancy", layout="wide")
st.title("🚀 AI Startup Consultancy Firm")

# Sidebar
with st.sidebar:
    st.header("About This App")
    st.write("Our AI board (CEO, CTO, CFO, COO) will analyze your business idea and provide comprehensive consultation.")
    
    st.markdown("---")
    st.subheader("How It Works")
    st.write("1. **Submit Idea:** Enter your business concept")
    st.write("2. **Board Discussion:** Watch AI agents collaborate")  
    st.write("3. **Final Report:** Get a comprehensive business analysis")
    st.write("4. **Export:** Download your report as PDF or Markdown")
    
    st.markdown("---")
    st.caption("Powered by LangGraph • OpenAI • Streamlit")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "👋 Welcome to your AI Startup Consultancy! I'm here to facilitate a board meeting with our expert AI agents.\n\n**Ready to analyze your business idea?** Our team includes:\n- 💼 **Sarah (CEO)**: Strategic vision and market analysis\n- 💻 **Mike (CTO)**: Technical feasibility and architecture\n- 💰 **Jennifer (CFO)**: Financial modeling and funding strategy\n- ⚙️ **Tom (COO)**: Operations and execution planning\n\nPlease share your business idea below to begin the consultation!"
    })

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    # Determine display name and avatar
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
    # Validate input
    if len(prompt.strip()) < 10:
        st.warning("Please provide a more detailed business idea (at least 10 characters)")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("You", avatar=AVATAR_ICONS["user"]):
        st.markdown(prompt)
    
    # Prepare initial message
    initial_messages = [HumanMessage(content=f"Analyze this business idea and provide comprehensive consultation: {prompt}")]
    
    # Process with spinner
    with st.spinner("🤝 The AI board is in session, analyzing your idea..."):
        try:
            # Stream from LangGraph with increased recursion limit
            config = {"recursion_limit": 50}  # Add this line
            for output in app.stream({"messages": initial_messages}, config=config):  # Add config parameter
                for key, value in output.items():
                    if key != "__end__" and "messages" in value and value["messages"]:
                        agent_role = key.lower()
                        agent_message = value['messages'][-1].content
                        
                        # Determine display info
                        names = {"ceo": "Sarah (CEO)", "cto": "Mike (CTO)", "cfo": "Jennifer (CFO)", "coo": "Tom (COO)"}
                        display_name = names.get(agent_role, key.upper())
                        avatar = AVATAR_ICONS.get(agent_role, "💬")
                        
                        # Display message
                        with st.chat_message(display_name, avatar=avatar):
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
        st.subheader("📄 Export Your Consultation Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_link = create_pdf_download_link(final_report, "startup_consultation_report")
                    if pdf_link:
                        st.markdown(pdf_link, unsafe_allow_html=True)
                        st.success("PDF download link generated!")
                    else:
                        st.error("Failed to generate PDF. Try downloading as Markdown instead.")
        
        with col2:
            # Clean content for markdown download as well
            clean_markdown = clean_text_for_pdf(final_report)
            st.download_button(
                label="📝 Download Markdown",
                data=clean_markdown,
                file_name="startup_consultation_report.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col3:
            if st.button("🔄 New Consultation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
    
    elif len(st.session_state.messages) > 15:
        st.markdown("---")
        st.info("💡 The consultation is in progress. The export options will appear once the final report is ready.")
        
        if st.button("🔄 Start New Consultation"):
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 12px;'>
    🤖 Powered by Multi-Agent AI • Built with LangGraph & Streamlit<br>
    For demonstration purposes • Results may not reflect real market conditions
    </div>
    """, 
    unsafe_allow_html=True
)
