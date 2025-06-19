import streamlit as st
import base64
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from main_v6 import app, RAG_AVAILABLE
from langchain_core.messages import HumanMessage
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import os

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

# DYNAMIC PATH RESOLUTION - WORKS ANYWHERE
def get_project_paths():
    """Get project paths that work regardless of where Streamlit is run from"""
    script_dir = Path(__file__).parent.absolute()
    
    # Check if we're in src directory or parent directory
    if script_dir.name == 'src':
        # Running from src directory
        src_dir = script_dir
        project_root = script_dir.parent
    else:
        # Running from parent directory
        src_dir = script_dir / 'src'
        project_root = script_dir
    
    return {
        'script_dir': script_dir,
        'src_dir': src_dir,
        'project_root': project_root,
        'knowledge_system': src_dir / 'knowledge_system',
        'data_sources': src_dir / 'knowledge_system' / 'data_sources',
        'scripts': src_dir / 'knowledge_system' / 'scripts'
    }

# Get paths once at module level
PATHS = get_project_paths()

def clean_text_for_pdf(text):
    """Clean text to remove Unicode characters that can't be encoded in latin-1"""
    replacements = {
        '\u2022': '•',
        '\u2013': '-',
        '\u2014': '--',
        '\u2018': "'",
        '\u2019': "'",
        '\u201c': '"',
        '\u201d': '"',
        '\u2026': '...',
        '\u00a0': ' '
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

def run_script_safely(script_name, description):
    """Run scripts with proper path resolution"""
    try:
        script_path = PATHS['scripts'] / script_name
        
        if not script_path.exists():
            st.error(f"❌ Script not found: {script_path}")
            return False
        
        # Run from project root with proper working directory
        result = subprocess.run([
            sys.executable, 
            str(script_path)
        ], capture_output=True, text=True, cwd=str(PATHS['project_root']))
        
        if result.returncode == 0:
            st.success(f"✅ {description} completed!")
            return True
        else:
            st.error(f"❌ {description} failed:")
            if result.stderr:
                st.code(result.stderr)
            if result.stdout:
                st.code(result.stdout)
            return False
            
    except Exception as e:
        st.error(f"❌ Error running {script_name}: {str(e)}")
        st.code(f"Paths used:\nScript: {PATHS['scripts'] / script_name}\nWorking dir: {PATHS['project_root']}")
        return False

# Set page config
st.set_page_config(page_title="RAG-Enhanced AI Startup Consultancy", layout="wide")

# Dynamic title based on RAG availability
if RAG_AVAILABLE:
    st.title("🚀 RAG-Enhanced AI Startup Consultancy Firm")
    st.markdown("*Powered by Knowledge Bases + Real-time Search*")
else:
    st.title("🚀 AI Startup Consultancy Firm")
    st.markdown("*Powered by Real-time Search*")

# Enhanced Sidebar with Data Ingestion Status
with st.sidebar:
    st.header("About This App")
    if RAG_AVAILABLE:
        st.write("Our AI board (CEO, CTO, CFO, COO) will analyze your business idea using **RAG-enhanced knowledge bases**, **real-time data feeds**, and **live market research**.")
    else:
        st.write("Our AI board (CEO, CTO, CFO, COO) will analyze your business idea using real-time market data.")
    
    st.markdown("---")
    st.subheader("How It Works")
    st.write("1. **Submit Idea:** Enter your business concept")
    st.write("2. **Board Discussion:** Watch AI agents collaborate")
    if RAG_AVAILABLE:
        st.write("3. **RAG Enhancement:** Agents access knowledge bases")
        st.write("4. **Live Data:** Real-time market feeds integrated")
        st.write("5. **Final Report:** Get comprehensive analysis")
        st.write("6. **Export:** Download as PDF or Markdown")
    else:
        st.write("3. **Final Report:** Get comprehensive analysis")
        st.write("4. **Export:** Download as PDF or Markdown")
    
    st.markdown("---")
    
    # Enhanced System Status
    st.subheader("System Status")
    if RAG_AVAILABLE:
        st.success("🔍 RAG System: Active")
        st.info("📚 Knowledge Bases: Loaded")
    else:
        st.warning("🔍 RAG System: Not Available")
    st.info("🌐 TavilySearch: Active")
    
    # Debug info
    with st.expander("🔧 Path Debug Info"):
        st.write(f"**Current working dir:** `{os.getcwd()}`")
        st.write(f"**Script dir:** `{PATHS['script_dir']}`")
        st.write(f"**Src dir:** `{PATHS['src_dir']}`")
        st.write(f"**Project root:** `{PATHS['project_root']}`")
        st.write(f"**Data sources:** `{PATHS['data_sources']}`")
        st.write(f"**Scripts:** `{PATHS['scripts']}`")
        
        # Check if paths exist
        st.write("**Path Status:**")
        st.write(f"- Data sources exist: {PATHS['data_sources'].exists()}")
        st.write(f"- Scripts exist: {PATHS['scripts'].exists()}")
        st.write(f"- data_ingestion.py: {(PATHS['scripts'] / 'data_ingestion.py').exists()}")
        st.write(f"- setup_knowledge_bases.py: {(PATHS['scripts'] / 'setup_knowledge_bases.py').exists()}")
    
    # Data Sources Status Section
    st.markdown("---")
    st.subheader("Data Sources Status")
    
    total_files = 0
    
    # Check for recent data files
    if PATHS['data_sources'].exists():
        agent_types = {
            "market_data": {"emoji": "📊", "name": "Market Intelligence"},
            "funding_data": {"emoji": "💰", "name": "Funding Data"},
            "tech_data": {"emoji": "💻", "name": "Tech Trends"},
            "operations_data": {"emoji": "⚙️", "name": "Operations Data"}
        }
        
        for agent_type, info in agent_types.items():
            agent_path = PATHS['data_sources'] / agent_type
            if agent_path.exists():
                files = list(agent_path.glob("*.txt"))
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    age = datetime.now() - datetime.fromtimestamp(latest_file.stat().st_mtime)
                    
                    if age.days < 1:
                        if age.seconds < 3600:  # Less than 1 hour
                            st.success(f"{info['emoji']} {info['name']}: Fresh ({age.seconds//60}m ago)")
                        else:  # Less than 1 day
                            st.success(f"{info['emoji']} {info['name']}: Fresh ({age.seconds//3600}h ago)")
                    elif age.days < 7:
                        st.warning(f"{info['emoji']} {info['name']}: {age.days} days old")
                    else:
                        st.error(f"{info['emoji']} {info['name']}: {age.days} days old")
                else:
                    st.error(f"{info['emoji']} {info['name']}: No data")
            else:
                st.error(f"{info['emoji']} {info['name']}: Missing")
        
        # Show total articles count
        total_files = sum(len(list((PATHS['data_sources'] / agent_type).glob("*.txt"))) 
                         for agent_type in agent_types.keys() 
                         if (PATHS['data_sources'] / agent_type).exists())
        
        if total_files > 0:
            st.metric("📚 Total Data Sources", f"{total_files} files")
        else:
            st.metric("📚 Total Data Sources", "No external data")
    else:
        st.warning("📁 Data sources directory not found")
    
    # Data Ingestion Controls
    st.markdown("---")
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Data", use_container_width=True, help="Fetch latest articles from RSS feeds"):
            with st.spinner("Fetching latest data..."):
                if run_script_safely("data_ingestion.py", "Data refresh"):
                    st.rerun()  # Refresh the page to show updated status
    
    with col2:
        if st.button("🔨 Rebuild KB", use_container_width=True, help="Rebuild knowledge bases with new data"):
            with st.spinner("Rebuilding knowledge bases..."):
                if run_script_safely("setup_knowledge_bases.py", "Knowledge base rebuild"):
                    st.rerun()
    
    # Real-time Feed Status
    st.markdown("---")
    st.subheader("Live Feeds")
    
    # Check for recent urgent updates
    urgent_files = []
    if PATHS['data_sources'].exists():
        for agent_type in ["market_data", "funding_data", "tech_data", "operations_data"]:
            agent_path = PATHS['data_sources'] / agent_type
            if agent_path.exists():
                urgent_files.extend(list(agent_path.glob("*urgent*.txt")))
    
    if urgent_files:
        # Find most recent urgent update
        latest_urgent = max(urgent_files, key=lambda x: x.stat().st_mtime)
        age = datetime.now() - datetime.fromtimestamp(latest_urgent.stat().st_mtime)
        if age.days < 1:
            st.success(f"🚨 Breaking News: {age.seconds//60}m ago")
        else:
            st.info("📡 Real-time feeds: Monitoring")
    else:
        st.info("📡 Real-time feeds: Monitoring")
    
    # Feed Sources
    with st.expander("📡 Data Sources", expanded=False):
        st.write("**Market Intelligence:**")
        st.write("• TechCrunch Startups")
        st.write("• Bloomberg Startup News")
        st.write("• Crunchbase News")
        
        st.write("**Funding Data:**")
        st.write("• Crunchbase Funding")
        st.write("• VentureBeat Business")
        
        st.write("**Technology Trends:**")
        st.write("• Stack Overflow Blog")
        st.write("• GitHub Blog")
        
        st.write("**Operations Insights:**")
        st.write("• Harvard Business Review")
    
    st.markdown("---")
    st.caption("Powered by LangGraph • OpenAI • RSS Feeds • Streamlit")
    
    # Enhanced status indicator
    if RAG_AVAILABLE and total_files > 0:
        st.success("🚀 **FULLY ENHANCED** - RAG + Live Data Active")
    elif RAG_AVAILABLE:
        st.warning("🔶 **PARTIALLY ENHANCED** - RAG Active, Add External Data")
    else:
        st.info("🔷 **BASIC MODE** - Real-time Search Only")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
    welcome_message = """👋 Welcome to your AI Startup Consultancy!

I'm here to facilitate a board meeting with our expert AI agents.

**Ready to analyze your business idea?**

Our team includes:
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
        '<div style="text-align: center; color: #666;">Powered by LangGraph • OpenAI • RAG Knowledge Bases • Real-time Feeds</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div style="text-align: center; color: #666;">Powered by LangGraph • OpenAI • TavilySearch • Real-time Market Data</div>',
        unsafe_allow_html=True
    )
