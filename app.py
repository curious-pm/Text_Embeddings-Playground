import streamlit as st
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Text Embedding Playground",
    page_icon="🔤",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metrics-container {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load API credentials securely from Streamlit secrets
API_KEY = st.secrets["azure_openai"]["api_key"]
TARGET_URL = st.secrets["azure_openai"]["target_url"]  # Complete target URL loaded from secrets

# Sidebar with educational content
with st.sidebar:
    st.title("📚 Learn About Text Embeddings")
    
    st.subheader("What are Text Embeddings?")
    st.write("""
    Text embeddings are numerical representations of text that capture semantic meaning. 
    They convert words and sentences into vectors of numbers, making it possible for 
    computers to understand and compare text similarity.
    """)
    
    st.subheader("How to Use This Tool")
    st.markdown("""
    1. **Predefined Phrases**: Enter reference phrases that will be used for comparison
    2. **Single Text Analysis**: Input a word/sentence to find similar phrases
    3. **Multi-Text Analysis**: Compare multiple texts and visualize their relationships
    """)

    st.subheader("Applications")
    st.markdown("""
    - Semantic search
    - Content recommendation
    - Text classification
    - Document similarity
    - Question answering
    """)

    st.subheader("Sample Inputs")
    st.markdown("**Reference Phrases:**")
    st.code("""
    - The weather is nice today
    - I love programming
    - Artificial intelligence is fascinating
    - ChatGPT is a powerful AI tool
    - Streamlit makes data visualization easy
    """, language="markdown")

    st.markdown("**Single Text Analysis Input:**")
    st.code("AI is revolutionizing the world", language="markdown")

    st.markdown("**Multi-Text Analysis Inputs:**")
    st.code("""
    - Machine learning is a subset of AI
    - I enjoy building applications using Streamlit
    - Programming in Python is fun and efficient
    - OpenAI is a leader in generative AI
    - Visualization tools make data analysis better
    """, language="markdown")
    st.markdown("---")
    st.markdown(
        """
        **🔗 GitHub Repository:**  
        [View Source Code](https://github.com/your-repo-name)
        """
         )

# Function to get text embeddings from Azure OpenAI
def get_text_embeddings(texts):
    if not texts or not any(text.strip() for text in texts):
        st.error("⚠️ Please enter some text to analyze.")
        return None

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }
    valid_texts = [text.strip() for text in texts if text.strip()]
    data = {"input": valid_texts}
    
    try:
        response = requests.post(TARGET_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        return [item["embedding"] for item in response.json()["data"]]
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network or API error: {str(e)}")
        return None

# Function to load predefined phrases
def load_predefined_phrases():
    st.markdown("  Enter Reference Phrases")
    predefined_phrases = st.text_area(
        label="Reference Phrases",
        label_visibility="collapsed",
        placeholder="Example:\nThe weather is nice today\nI love programming\nArtificial intelligence is fascinating",
        height=150
    )
    return predefined_phrases.split("\n") if predefined_phrases.strip() else []

# Function to construct a meaningful sentence from embeddings
def construct_meaningful_sentence(input_text, predefined_phrases, num_similar=3):
    if not predefined_phrases:
        st.warning("⚠️ Please provide predefined phrases first.")
        return None

    embeddings = get_text_embeddings([input_text] + predefined_phrases)
    if not embeddings:
        return None

    input_embedding = embeddings[0]
    predefined_embeddings = embeddings[1:]
    similarities = cosine_similarity([input_embedding], predefined_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:num_similar]
    top_phrases = [predefined_phrases[idx] for idx in top_indices]
    top_scores = [similarities[idx] for idx in top_indices]
    
    return top_phrases, top_scores

# Function to create a similarity heatmap
def create_similarity_heatmap(similarity_matrix, labels):
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=labels,
        y=labels,
        colorscale="Viridis"
    ))
    fig.update_layout(title="Similarity Heatmap", height=500, width=700)
    return fig

# Function to create an interactive scatter plot
def create_interactive_scatter(reduced_embeddings, labels):
    df = pd.DataFrame(reduced_embeddings, columns=['PC1', 'PC2'])
    df['text'] = labels

    fig = px.scatter(
        df, x='PC1', y='PC2', text='text',
        title="Interactive 2D Visualization of Embeddings"
    )
    fig.update_traces(marker=dict(size=12), textposition='top center')
    fig.update_layout(height=500, width=700)
    return fig

# Main application layout
st.title("Text Embedding Playground")

st.markdown("Explore the power of text embeddings through interactive analysis and visualization")

# Tabs for functionality
tab1, tab2 = st.tabs(["📊 Single Text Analysis", "🔍 Multi-Text Analysis"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        predefined_phrases = load_predefined_phrases()
        if st.button("💾 Save Reference Phrases"):
            if predefined_phrases:
                st.success("✅ Reference phrases saved successfully!")
            else:
                st.warning("⚠️ Please enter some phrases first")
    
    with col2:
        st.markdown(
        """
        <div>
            Enter text to analyze:
        </div>
        """,
        unsafe_allow_html=True,)
        user_input = st.text_input("", placeholder="Type your text here...")
        if st.button("🔄 Analyze"):
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to analyze")
            elif not predefined_phrases:
                st.warning("⚠️ Please add some reference phrases first")
            else:
                with st.spinner("🔄 Processing your input..."):
                    try:
                        similar_phrases, similarity_scores = construct_meaningful_sentence(user_input, predefined_phrases)
                        st.markdown("### 📊 Analysis Results")
                        for phrase, score in zip(similar_phrases, similarity_scores):
                            st.markdown(f"""
                            <div class="metrics-container">
                                <h4>{phrase}</h4>
                                <p>Similarity Score: {score:.4f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")

with tab2:
    multi_text_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="Example:\nFirst text here\nSecond text here\nThird text here",
        height=150
    )
    if st.button("🔄 Analyze Texts"):
        texts = [t.strip() for t in multi_text_input.split("\n") if t.strip()]
        if len(texts) < 2:
            st.warning("⚠️ Please enter at least 2 texts for comparison")
        else:
            with st.spinner("🔄 Processing texts..."):
                try:
                    embeddings = get_text_embeddings(texts)
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(create_similarity_heatmap(similarity_matrix, texts))
                    with col2:
                        reduced_embeddings = PCA(n_components=2).fit_transform(embeddings)
                        st.plotly_chart(create_interactive_scatter(reduced_embeddings, texts))
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")

st.markdown(
    """
    [Visit Organization @Curios-PM](https://nas.io/curious-pm)
    """
)
