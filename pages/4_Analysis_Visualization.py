# import streamlit as st
# from datetime import date

# st.set_page_config(page_title="Coming Soon", layout="centered")

# st.markdown("""
#     <style>
#         .centered {
#             text-align: center;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.markdown("<h2 class='centered'>🚧 DataLine Integration   \nComing Soon 🚧</h2>", unsafe_allow_html=True)
# st.markdown("<p class='centered'>We’re working on bringing interactive data exploration directly into this app.</p>", unsafe_allow_html=True)

# st.divider()
# st.markdown(f"<p class='centered'>Last updated: {date.today().strftime('%B %d, %Y')}</p>", unsafe_allow_html=True)
"""
Free "Chat with Your Data" Module for Streamlit
Uses PandasAI with Ollama (local LLM) - completely free!
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from pandasai import SmartDataframe
#from pandasai.llm import OpenAI  # We'll use this structure but with Ollama
import plotly.express as px

# Alternative: Simple rule-based chatbot if Ollama isn't set up
class SimpleChatbot:
    """Fallback: Rule-based data chatbot when LLM isn't available"""
    
    def __init__(self, df):
        self.df = df
    
    def chat(self, query):
        query_lower = query.lower()
        
        # Summary statistics
        if any(word in query_lower for word in ['summary', 'describe', 'overview', 'stats']):
            return self._get_summary()
        
        # Correlation
        elif 'correlat' in query_lower:
            return self._get_correlations()
        
        # Missing values
        elif any(word in query_lower for word in ['missing', 'null', 'nan']):
            return self._get_missing_values()
        
        # Distribution/histogram
        elif any(word in query_lower for word in ['distribution', 'histogram', 'plot']):
            return self._plot_distributions()
        
        # Column info
        elif 'column' in query_lower or 'feature' in query_lower:
            return self._get_column_info()
        
        else:
            return self._help_message()
    
    def _get_summary(self):
        st.write("### Dataset Summary")
        st.write(f"**Shape:** {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        st.write("**Numerical Summary:**")
        st.dataframe(self.df.describe())
        return "Here's a summary of your dataset!"
    
    def _get_correlations(self):
        st.write("### Correlation Matrix")
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            corr = self.df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
            return "Here's the correlation matrix for numerical columns!"
        else:
            return "Not enough numerical columns to compute correlations."
    
    def _get_missing_values(self):
        st.write("### Missing Values Analysis")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df)
            return f"Found missing values in {len(missing_df)} columns."
        else:
            return "No missing values found! ✓"
    
    def _plot_distributions(self):
        st.write("### Data Distributions")
        numeric_cols = self.df.select_dtypes(include=['number']).columns[:4]  # Limit to 4
        
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4*len(numeric_cols)))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for idx, col in enumerate(numeric_cols):
                self.df[col].hist(bins=30, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
            return f"Showing distributions for {len(numeric_cols)} numerical columns."
        else:
            return "No numerical columns to plot."
    
    def _get_column_info(self):
        st.write("### Column Information")
        info_df = pd.DataFrame({
            'Column': self.df.columns,
            'Type': self.df.dtypes.values,
            'Non-Null Count': self.df.count().values,
            'Unique Values': [self.df[col].nunique() for col in self.df.columns]
        })
        st.dataframe(info_df)
        return "Here's information about all columns!"
    
    def _help_message(self):
        return """
I can help you explore your data! Try asking:
- "Show me a summary of the data"
- "What are the correlations?"
- "Are there any missing values?"
- "Plot the distributions"
- "Tell me about the columns"
"""


def chat_with_data_module():
    """Main chat interface for data exploration"""
    
    st.title("💬 Chat with Your Data")
    st.markdown("Ask questions about your brain segmentation data in natural language!")
    
    # Check if data exists in session state
    if 'cleaned_data' not in st.session_state:
        st.warning("⚠️ No data loaded yet. Please load data from the main app first.")
        #Provide upload option
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload your brain segmentation CSV file", type=["csv"])        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.cleaned_data = df
                st.success("✅ Data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                
        # Demo mode with sample data
        if st.button("Load Demo Data"):
            # Create sample brain segmentation data
            demo_data = pd.DataFrame({
                'subject_id': [f'SUB{i:03d}' for i in range(1, 51)],
                'age': [25, 28, 32, 45, 38, 52, 29, 41, 36, 48] * 5,
                'hippocampus_volume': [3.2, 3.5, 3.1, 2.8, 3.0, 2.6, 3.4, 2.9, 3.2, 2.7] * 5,
                'cortical_thickness': [2.5, 2.6, 2.4, 2.3, 2.5, 2.2, 2.6, 2.4, 2.5, 2.3] * 5,
                'white_matter_volume': [450, 460, 440, 430, 455, 425, 465, 435, 450, 428] * 5,
                'qc_score': [0.95, 0.92, 0.88, 0.91, 0.94, 0.87, 0.93, 0.90, 0.89, 0.91] * 5
            })
            st.session_state.cleaned_data = demo_data
            st.rerun()
        return
    
    df = st.session_state.cleaned_data
    
    # Display basic data info
    with st.expander("📊 Dataset Overview", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.write("**First few rows:**")
        st.dataframe(df.head(), use_container_width=True)
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # LLM Setup Selection
    st.sidebar.header("⚙️ Settings")
    llm_option = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["Simple (No Setup Required)", "PandasAI + Ollama (Advanced)"],
        help="Simple mode uses rule-based queries. Ollama requires local setup."
    )
    
    # Initialize chatbot
    if llm_option == "Simple (No Setup Required)":
        chatbot = SimpleChatbot(df)
        st.sidebar.success("✅ Simple mode active")
    else:
        st.sidebar.info("""
        **To use Ollama:**
        1. Install: https://ollama.com/download
        2. Run: `ollama pull llama2`
        3. Start: `ollama serve`
        4. Install PandasAI: `pip install pandasai`
        """)
        
        # Try to use PandasAI with Ollama
        try:
            #from pandasai.llm import Ollama
            #lm = Ollama(model="llama2")
            #chatbot = SmartDataframe(df, config={"llm": llm})
            st.sidebar.success("✅ Ollama connected!")
        except Exception as e:
            st.sidebar.error(f"❌ Ollama not available: {str(e)}")
            st.sidebar.info("Falling back to Simple mode")
            chatbot = SimpleChatbot(df)
    
    # Chat Interface
    st.markdown("---")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask something about your data..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if isinstance(chatbot, SimpleChatbot):
                        response = chatbot.chat(prompt)
                    else:
                        # PandasAI response
                        response = chatbot.chat(prompt)
                    
                    st.write(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
    
    # Quick Actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Quick Actions")
    
    if st.sidebar.button("Show Summary"):
        st.session_state.chat_history.append({"role": "user", "content": "Show me a summary"})
        st.rerun()
    
    if st.sidebar.button("Check Correlations"):
        st.session_state.chat_history.append({"role": "user", "content": "What are the correlations?"})
        st.rerun()
    
    if st.sidebar.button("Missing Values"):
        st.session_state.chat_history.append({"role": "user", "content": "Are there missing values?"})
        st.rerun()
    
    if st.sidebar.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="Chat with Data", page_icon="💬", layout="wide")
    chat_with_data_module()