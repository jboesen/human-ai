import streamlit as st
import openai
from openai import OpenAI
import tiktoken
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import io
import sys
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from contextlib import redirect_stdout, redirect_stderr
import re
import json
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
import difflib

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ChatManager:
    """Manages multiple chat sessions with search and storage capabilities"""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state for chat management"""
        if 'chats' not in st.session_state:
            st.session_state.chats = {}
        if 'current_chat_id' not in st.session_state:
            st.session_state.current_chat_id = None
        if 'chat_search_query' not in st.session_state:
            st.session_state.chat_search_query = ""
        if 'show_chat_popover' not in st.session_state:
            st.session_state.show_chat_popover = False
    
    def create_new_chat(self, title: str = None) -> str:
        """Create a new chat session"""
        chat_id = str(uuid.uuid4())
        if not title:
            title = f"Chat {len(st.session_state.chats) + 1}"
        
        st.session_state.chats[chat_id] = {
            'id': chat_id,
            'title': title,
            'messages': [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        st.session_state.current_chat_id = chat_id
        return chat_id
    
    def get_current_chat(self) -> Dict:
        """Get the current active chat"""
        if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chats:
            # Create first chat if none exists
            self.create_new_chat("New Chat")
        return st.session_state.chats[st.session_state.current_chat_id]
    
    def switch_to_chat(self, chat_id: str):
        """Switch to a different chat"""
        if chat_id in st.session_state.chats:
            st.session_state.current_chat_id = chat_id
            st.session_state.show_chat_popover = False
    
    def update_chat_title(self, chat_id: str, title: str):
        """Update the title of a chat"""
        if chat_id in st.session_state.chats:
            st.session_state.chats[chat_id]['title'] = title
            st.session_state.chats[chat_id]['updated_at'] = datetime.now().isoformat()
    
    def delete_chat(self, chat_id: str):
        """Delete a chat session"""
        if chat_id in st.session_state.chats:
            del st.session_state.chats[chat_id]
            if st.session_state.current_chat_id == chat_id:
                # Switch to another chat or create new one
                if st.session_state.chats:
                    st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                else:
                    self.create_new_chat()
    
    def add_message_to_current_chat(self, role: str, content: str):
        """Add a message to the current chat"""
        current_chat = self.get_current_chat()
        current_chat['messages'].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        current_chat['updated_at'] = datetime.now().isoformat()
        
        # Auto-update title based on first user message
        if role == 'user' and len(current_chat['messages']) == 1:
            title = content[:50] + "..." if len(content) > 50 else content
            self.update_chat_title(current_chat['id'], title)
    
    def search_chats(self, query: str, limit: int = 3) -> List[Dict]:
        """Search through chats and return top results by relevance + recency"""
        if not query.strip():
            # Return most recent chats
            chats = list(st.session_state.chats.values())
            chats.sort(key=lambda x: x['updated_at'], reverse=True)
            return chats[:limit]
        
        query_lower = query.lower()
        scored_chats = []
        
        for chat in st.session_state.chats.values():
            relevance_score = 0
            
            # Search in title
            title_match = difflib.SequenceMatcher(None, query_lower, chat['title'].lower()).ratio()
            relevance_score += title_match * 3  # Weight title matches higher
            
            # Search in message content
            message_content = " ".join([msg['content'] for msg in chat['messages']])
            content_match = difflib.SequenceMatcher(None, query_lower, message_content.lower()).ratio()
            relevance_score += content_match
            
            # Check for exact word matches (bonus points)
            query_words = query_lower.split()
            for word in query_words:
                if word in chat['title'].lower():
                    relevance_score += 2
                if word in message_content.lower():
                    relevance_score += 1
            
            # Recency boost (newer chats get slight preference)
            try:
                updated_time = datetime.fromisoformat(chat['updated_at'])
                hours_old = (datetime.now() - updated_time).total_seconds() / 3600
                recency_boost = max(0, 1 - (hours_old / (24 * 7)))  # Boost fades over a week
                relevance_score += recency_boost * 0.5
            except:
                pass
            
            if relevance_score > 0:
                scored_chats.append((relevance_score, chat))
        
        # Sort by score and return top results
        scored_chats.sort(key=lambda x: x[0], reverse=True)
        return [chat for score, chat in scored_chats[:limit]]
    
    def export_chats(self) -> str:
        """Export all chats to JSON string"""
        return json.dumps(st.session_state.chats, indent=2)
    
    def import_chats(self, json_data: str):
        """Import chats from JSON string"""
        try:
            imported_chats = json.loads(json_data)
            st.session_state.chats.update(imported_chats)
            return True
        except:
            return False

def init_openai():
    api_key = st.session_state.get('openai_api_key')
    if api_key:
        return OpenAI(api_key=api_key)
    return None

def extract_code_blocks(text):
    """Extract Python code blocks from markdown text"""
    pattern = r'```(?:python)?\n?(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

def render_latex_markdown(text):
    """Convert LaTeX in text to HTML format for Streamlit rendering"""
    text = re.sub(r'\$([^$]+)\$', r'\\(\1\\)', text)
    text = re.sub(r'\$\$([^$]+)\$\$', r'\\[\1\\]', text)
    return text

def execute_python_code(code):
    """Execute Python code and capture output, errors, and plots"""
    output = io.StringIO()
    error = io.StringIO()
    plots = []
    
    plt.clf()
    
    try:
        namespace = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go,
            'st': st,
        }
        
        with redirect_stdout(output), redirect_stderr(error):
            exec(code, namespace)
            
        if plt.get_fignums():
            plots.append(plt.gcf())
            
    except Exception as e:
        error.write(f"Error: {str(e)}\n{traceback.format_exc()}")
    
    return output.getvalue(), error.getvalue(), plots

def get_adverb_ending_tokens(model="gpt-4o"):
    """Get tokens that end with common adverb patterns"""
    encoding = tiktoken.encoding_for_model(model)
    adverb_tokens = {}
    
    common_adverbs = [
        "significantly", "effectively", "efficiently", "strategically", "systematically",
        "comprehensively", "holistically", "specifically", "particularly", "essentially",
        "substantially", "considerably", "remarkably", "exceptionally", "optimally",
        "seamlessly", "innovatively", "dynamically", "proactively", "collaboratively"
    ]
    
    for adverb in common_adverbs:
        try:
            tokens = encoding.encode(adverb)
            tokens.extend(encoding.encode(f" {adverb}"))
            for token_id in tokens:
                adverb_tokens[str(token_id)] = -3
        except:
            continue
    
    return adverb_tokens

def get_token_bias_dict(banned_words, heavy_penalty=-25, model="gpt-4o"):
    """Convert banned words to token IDs with tiered penalties"""
    encoding = tiktoken.encoding_for_model(model)
    bias_dict = {}
    words_processed = []
    words_skipped = []
    
    adverb_bias = get_adverb_ending_tokens(model)
    bias_dict.update(adverb_bias)
    
    core_ai_words = [
        "leverage", "utilize", "optimize", "facilitate", "implement", "enhance",
        "streamline", "synergy", "paradigm", "framework", "methodology", 
        "infrastructure", "scalable", "robust", "seamless", "innovative",
        "cutting-edge", "state-of-the-art", "groundbreaking", "transformative",
        "dynamic", "comprehensive", "holistic", "strategic", "actionable",
        "deliverable", "best-practice", "deep-dive", "circle-back", "touch-base"
    ]
    
    for word in core_ai_words:
        if word in banned_words:
            variants = [word, f" {word}", word.lower(), f" {word.lower()}"]
            word_added = False
            for variant in variants:
                try:
                    tokens = encoding.encode(variant)
                    for token_id in tokens:
                        if len(bias_dict) >= 300:
                            words_skipped.extend(banned_words[banned_words.index(word):])
                            raise ValueError(f"Hit 300 token limit! Processed {len(words_processed)} words, skipped {len(words_skipped)} words")
                        bias_dict[str(token_id)] = heavy_penalty
                        word_added = True
                except ValueError:
                    raise
                except:
                    continue
            if word_added:
                words_processed.append(word)
    
    remaining_words = [w for w in banned_words if w not in core_ai_words]
    medium_penalty = heavy_penalty // 2
    
    for word in remaining_words:
        variants = [word, f" {word}", word.lower(), f" {word.lower()}"]
        word_added = False
        for variant in variants:
            try:
                tokens = encoding.encode(variant)
                for token_id in tokens:
                    if len(bias_dict) >= 300:
                        words_skipped.extend(remaining_words[remaining_words.index(word):])
                        raise ValueError(f"Hit 300 token limit! Processed {len(words_processed)} core + {len([w for w in words_processed if w in remaining_words])} other words")
                    bias_dict[str(token_id)] = medium_penalty
                    word_added = True
            except ValueError:
                raise
            except:
                continue
        if word_added:
            words_processed.append(word)
    
    return bias_dict, words_processed, words_skipped

def send_humanized_message(client, messages, banned_words, penalty, model):
    """Send message with banned words penalized via logit_bias"""
    try:
        logit_bias, words_processed, words_skipped = get_token_bias_dict(banned_words, penalty, model)
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            logit_bias=logit_bias,
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        return response, len(logit_bias), words_processed, words_skipped
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, [], []

def setup_mathjax():
    """Add MathJax support to the page"""
    mathjax_script = """
    <script>
    window.MathJax = {
      tex: {
        inlineMath: [['\\\\(', '\\\\)']],
        displayMath: [['\\\\[', '\\\\]']],
        processEscapes: true,
        processEnvironments: true
      },
      options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
      }
    };
    </script>
    <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """
    st.markdown(mathjax_script, unsafe_allow_html=True)

def render_chat_popover(chat_manager: ChatManager):
    """Render the chat search and management popover"""
    
    # Keyboard shortcut detection with better cross-platform support
    keyboard_script = """
    <script>
    document.addEventListener('keydown', function(e) {
        // Check for Cmd+K on Mac or Ctrl+K on Windows/Linux
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            // Find the search button and click it
            const searchBtn = document.querySelector('[data-testid*="search"]') || 
                             document.querySelector('button[title*="Search"]') ||
                             Array.from(document.querySelectorAll('button')).find(btn => 
                                 btn.textContent.includes('Search') || btn.textContent.includes('🔍')
                             );
            if (searchBtn) {
                searchBtn.click();
            }
        }
    });
    </script>
    """
    st.markdown(keyboard_script, unsafe_allow_html=True)
    
    # Popover modal with floating style
    if st.session_state.show_chat_popover:
        # Create floating modal overlay
        modal_css = """
        <style>
        .chat-modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding-top: 8vh;
            backdrop-filter: blur(4px);
        }
        .chat-modal {
            background: white;
            border-radius: 12px;
            padding: 24px;
            width: 90%;
            max-width: 650px;
            max-height: 75vh;
            overflow-y: auto;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
            animation: slideIn 0.2s ease-out;
        }
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0) scale(1);
            }
        }
        .stButton > button {
            border-radius: 8px;
        }
        </style>
        """
        st.markdown(modal_css, unsafe_allow_html=True)
        
        # Create a container that acts as the modal
        with st.container():
            # Modal content
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown("### 🔍 Search & Manage Chats")
            with col2:
                if st.button("✕", key="close_popover", help="Close (Esc)"):
                    st.session_state.show_chat_popover = False
                    st.rerun()
            
            # Search input
            search_query = st.text_input(
                "Search chats...", 
                value=st.session_state.chat_search_query,
                placeholder="Search by title or content",
                key="chat_search_input"
            )
            st.session_state.chat_search_query = search_query
            
            # Search results
            search_results = chat_manager.search_chats(search_query, limit=5)
            
            if search_results:
                st.markdown("#### Search Results")
                for chat in search_results:
                    col1, col2, col3 = st.columns([4, 1, 1])
                    
                    with col1:
                        # Chat title and preview
                        is_current = chat['id'] == st.session_state.current_chat_id
                        current_indicator = "🔸 " if is_current else ""
                        
                        if st.button(
                            f"{current_indicator}{chat['title']}", 
                            key=f"switch_to_{chat['id']}",
                            help=f"Created: {chat['created_at'][:16]}"
                        ):
                            chat_manager.switch_to_chat(chat['id'])
                            st.rerun()
                        
                        # Show preview of last message
                        if chat['messages']:
                            last_msg = chat['messages'][-1]['content'][:100]
                            st.caption(f"Last: {last_msg}...")
                    
                    with col2:
                        # Edit title button
                        if st.button("✏️", key=f"edit_{chat['id']}", help="Edit title"):
                            new_title = st.text_input(
                                "New title:", 
                                value=chat['title'],
                                key=f"title_input_{chat['id']}"
                            )
                            if st.button("Save", key=f"save_title_{chat['id']}"):
                                chat_manager.update_chat_title(chat['id'], new_title)
                                st.rerun()
                    
                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{chat['id']}", help="Delete chat"):
                            if len(st.session_state.chats) > 1:  # Keep at least one chat
                                chat_manager.delete_chat(chat['id'])
                                st.rerun()
                            else:
                                st.error("Cannot delete the last chat")
                    
                    st.divider()
            else:
                st.info("No chats found")

def main():
    st.set_page_config(page_title="AI Humanizer Chat with Multi-Chat Support", layout="wide")
    
    # Initialize chat manager
    chat_manager = ChatManager()
    
    # Add MathJax support
    setup_mathjax()
    
    # Header with minimal controls
    current_chat = chat_manager.get_current_chat()
    
    # Simple header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"🤖➡️👤 {current_chat['title']}")
    with col2:
        if st.button("🔍", key="open_search", help="Search Chats (⌘K / Ctrl+K)"):
            st.session_state.show_chat_popover = True
    
    # Render chat popover if active
    if st.session_state.show_chat_popover:
        render_chat_popover(chat_manager)
    
    # Sidebar for settings
    with st.sidebar:
        # Collapsible settings section
        with st.expander("⚙️ Settings", expanded=False):
            # API Key
            if 'openai_api_key' not in st.session_state:
                st.session_state.openai_api_key = ""
            
            api_key = st.text_input(
                "OpenAI API Key", 
                value=st.session_state.openai_api_key,
                type="password"
            )
            
            if api_key:
                st.session_state.openai_api_key = api_key
            
            # Model selection
            model = st.selectbox("Model", [
                "gpt-4o", 
                "gpt-4o-mini", 
                "gpt-4-turbo",
                "gpt-3.5-turbo"
            ])
            
            st.divider()
            
            # Code execution settings
            st.subheader("🐍 Code Execution")
            
            auto_execute = st.checkbox(
                "Auto-execute Python code blocks", 
                value=True,
                help="Automatically run Python code found in AI responses"
            )
            
            show_code_output = st.checkbox(
                "Show code output", 
                value=True,
                help="Display print statements and other output from code execution"
            )
            
            st.divider()
            
            # LaTeX settings
            st.subheader("📐 LaTeX Support")
            st.info("LaTeX is automatically rendered:\n- Inline: `$x^2$` → $x^2$\n- Block: `$$E=mc^2$$` → $$E=mc^2$$")
            
            st.divider()
            
            # Penalty slider
            penalty = st.slider(
                "Core AI Words Penalty", 
                min_value=0.0, 
                max_value=12.0, 
                value=10.0, 
                step=0.1,
                help="Penalty for worst AI-speak words"
            )
            
            st.divider()
            
            # Banned words management
            st.subheader("🚫 Banned AI Words")
            
            default_words = [
                "insight", "hone", "dissect", "resonate", "eager", "delve", "foster",
                "facilitate", "significantly", "inform", "potential", "multifaceted",
                "modern", "bridge", "suggest", "elevate", "leverage", "navigate",
                "evolve", "captivate", "utilize", "crucial", "paramount", "ultimately",
                "tapestry", "beacon", "symphony", "provide", "remain", "represent",
                "undergo", "rather", "appears", "demonstrate", "furthermore", "moreover",
                "comprehensive", "however", "therefore", "additionally", "subsequently",
                "nonetheless", "consequently", "particularly", "specifically", "essentially",
                "effectively", "efficiently", "optimize", "enhance", "implement", "robust",
                "seamless", "innovative", "cutting-edge", "state-of-the-art", "groundbreaking",
                "revolutionary", "transformative", "dynamic", "synergy", "paradigm",
                "methodology", "framework", "infrastructure", "scalable", "streamline",
                "orchestrate", "imperative", "encompasses", "exhibits", "vital",
                "interplay", "quintessential", "amalgamation", "pivotal", "nuanced",
                "intricate", "meticulous", "discerning", "profound", "conducive",
                "myriad", "plethora", "inherently", "exemplifies"
            ]
            default_words = sorted(set([w for w in default_words if w.strip()]))
            
            banned_words_text = st.text_area(
                "Edit banned words (one per line):",
                value="\n".join(default_words),
                height=200,
                help="These words will be penalized during generation"
            )
            banned_words = [word.strip() for word in banned_words_text.split('\n') if word.strip()]
            stemmer = PorterStemmer()
            banned_words = sorted(set([stemmer.stem(word) for word in banned_words]))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Words", len(banned_words))
            with col2:
                st.metric("Penalty", f"-{penalty}")
        
        st.divider()
        
        # Chat management section
        st.subheader("💬 Chat Management")
        
        # New chat button (moved to sidebar)
        if st.button("➕ New Chat", key="sidebar_new_chat", use_container_width=True):
            chat_manager.create_new_chat()
            st.rerun()
        
        st.metric("Total Chats", len(st.session_state.chats))
        current_title = current_chat['title'][:25] + "..." if len(current_chat['title']) > 25 else current_chat['title']
        st.metric("Current Chat", current_title)
        
        # Export/Import
        with st.expander("📤 Export/Import Chats"):
            if st.button("Export All Chats"):
                export_data = chat_manager.export_chats()
                st.download_button(
                    "Download JSON",
                    export_data,
                    file_name=f"ai_chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            uploaded_file = st.file_uploader("Import Chats", type=['json'])
            if uploaded_file is not None:
                import_data = uploaded_file.read().decode('utf-8')
                if chat_manager.import_chats(import_data):
                    st.success("Chats imported successfully!")
                    st.rerun()
                else:
                    st.error("Failed to import chats")
        
        st.divider()
        
        # Clear current chat button
        if st.button("🗑️ Clear Current Chat", use_container_width=True):
            current_chat['messages'] = []
            st.rerun()
    
    # Initialize client
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar settings")
        return
    
    client = init_openai()
    if not client:
        st.error("❌ Failed to initialize OpenAI client")
        return
    
    # Display chat messages from current chat
    for message in current_chat['messages']:
        with st.chat_message(message["role"]):
            rendered_content = render_latex_markdown(message["content"])
            st.markdown(rendered_content, unsafe_allow_html=True)
            
            if (message["role"] == "assistant" and auto_execute and 
                "```" in message["content"]):
                
                code_blocks = extract_code_blocks(message["content"])
                
                for i, code in enumerate(code_blocks):
                    if code.strip():
                        st.subheader(f"🐍 Code Block {i+1}")
                        st.code(code, language="python")
                        
                        with st.spinner("Running code..."):
                            output, error, plots = execute_python_code(code)
                        
                        if plots:
                            for j, plot in enumerate(plots):
                                st.pyplot(plot, clear_figure=True)
                        
                        if output and show_code_output:
                            st.subheader("📤 Output")
                            st.text(output)
                        
                        if error:
                            st.subheader("⚠️ Errors")
                            st.error(error)
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to current chat
        chat_manager.add_message_to_current_chat("user", prompt)
        
        with st.chat_message("user"):
            rendered_prompt = render_latex_markdown(prompt)
            st.markdown(rendered_prompt, unsafe_allow_html=True)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            system_prompt = """
            You write like a knowledgeable human having a conversation. Use simple, direct language. Skip business jargon and overly formal phrasing. Be authentic.

            When users ask for data analysis, calculations, or visualizations, provide Python code using these available libraries:
            - pandas (pd)
            - numpy (np) 
            - matplotlib.pyplot (plt)
            - seaborn (sns)
            - plotly.express (px)
            - plotly.graph_objects (go)

            For mathematical expressions, use LaTeX notation:
            - Inline math: $x^2 + y^2 = z^2$
            - Block math: $$\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}$$

            Writing guidelines:
            - Cut unnecessary adjectives and adverbs
            - Use fewer linking verbs, but when you must, use versions of "to be"
            - Choose concrete subjects and active verbs over abstract nouns
            - Use semicolons for closely-related ideas and colons when appropriate
            - When providing code, explain what it does in simple terms
            - Include sample data in your code examples when helpful
            - Use LaTeX for mathematical notation when appropriate
            """
            
            # Prepare messages for API
            api_messages = [{"role": "system", "content": system_prompt}]
            api_messages.extend([{
                "role": msg["role"], 
                "content": msg["content"]
            } for msg in current_chat['messages']])
            
            # Stream response with logit bias
            response_stream, tokens_biased, words_processed, words_skipped = send_humanized_message(
                client, api_messages, banned_words, penalty, model
            )
            
            if response_stream:
                if tokens_biased > 0:
                    st.caption(f"🎯 Applied tiered penalties: {len(words_processed)} words, {tokens_biased} tokens")
                    if words_skipped:
                        st.caption(f"⚠️ Skipped {len(words_skipped)} words due to token limit")
                
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        rendered_response = render_latex_markdown(full_response + "▌")
                        message_placeholder.markdown(rendered_response, unsafe_allow_html=True)
                
                # Final render without cursor
                rendered_final = render_latex_markdown(full_response)
                message_placeholder.markdown(rendered_final, unsafe_allow_html=True)
                
                # Add assistant response to current chat
                chat_manager.add_message_to_current_chat("assistant", full_response)
                
                # Auto-execute code if enabled
                if auto_execute and "```" in full_response:
                    code_blocks = extract_code_blocks(full_response)
                    
                    for i, code in enumerate(code_blocks):
                        if code.strip():
                            st.subheader(f"🐍 Code Block {i+1}")
                            st.code(code, language="python")
                            
                            with st.spinner("Running code..."):
                                output, error, plots = execute_python_code(code)
                            
                            if plots:
                                for j, plot in enumerate(plots):
                                    st.pyplot(plot, clear_figure=True)
                            
                            if output and show_code_output:
                                st.subheader("📤 Output")
                                st.text(output)
                            
                            if error:
                                st.subheader("⚠️ Errors")
                                st.error(error)
                    
            else:
                st.error("❌ Failed to get response from API")

if __name__ == "__main__":
    main()
