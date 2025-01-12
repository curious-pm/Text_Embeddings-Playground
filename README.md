# **Text Embedding Playground**

Explore the power of text embeddings through an interactive application that enables semantic analysis and visualization of text relationships.

## **Introduction**

This application demonstrates how text embeddings, generated using Azure OpenAI, can be used to perform tasks such as:
- Analyzing similarity between texts
- Visualizing text relationships in 2D space
- Supporting applications like semantic search, recommendation systems, and more.

It leverages **Streamlit** for the user interface, **Azure OpenAI's text-embedding-ada-002** for embedding generation, and tools like **scikit-learn** and **Plotly** for visualization.

---

## **What are Text Embeddings?**

**Text embeddings** are vectorized numerical representations of text that capture the semantic meaning of words, sentences, or paragraphs. They allow computers to:
- Compare text similarity.
- Perform clustering, classification, and information retrieval tasks.

For example:
- The phrase "I love programming" will have a vector close to "Coding is fun" but far from "It's raining outside."

Text embeddings are widely used in natural language processing (NLP) applications such as:
- Semantic search engines
- Recommendation systems
- Document similarity detection
- Question-answering systems

---

## **Applications of Text Embeddings**

Here are some real-world use cases:

1. **Semantic Search**: Searching for similar documents or responses by analyzing their embeddings instead of just keywords.
2. **Content Recommendation**: Recommending similar articles, videos, or products based on text descriptions.
3. **Text Classification**: Categorizing texts (e.g., spam detection, sentiment analysis).
4. **Question Answering**: Identifying the most relevant answer to a question using similarity scores.

---

## **How This Application Works**

### **High-Level Workflow**
1. **Input**: Users provide reference phrases or input texts for analysis.
2. **Embedding Generation**: The application uses Azure OpenAI to convert the input text into embeddings (vector representations).
3. **Similarity Calculation**: Cosine similarity is calculated to determine how closely related two embeddings are.
4. **Visualization**: The relationships between embeddings are visualized using:
   - Heatmaps (similarity matrix)
   - 2D scatter plots (dimensionality reduction via PCA)

---

## **Code Flow**

### **1. Loading Secrets**
The application loads the `API_KEY` and `TARGET_URL` from the `secrets.toml` file. These credentials are essential for connecting to Azure OpenAI.

### **2. Sidebar**
The sidebar provides:
- Educational content on embeddings.
- Sample inputs to guide the user.
- A link to the GitHub repository.

### **3. Input Handling**
Users can:
- Enter **reference phrases** (for comparison).
- Enter **single text** (to find similarities).
- Enter **multiple texts** (to analyze relationships between texts).

### **4. Embedding Generation**
The `get_text_embeddings` function sends a request to the Azure OpenAI API to generate embeddings for the provided texts.

### **5. Analysis and Visualization**
- **Cosine Similarity**: Measures the similarity between vectors.
- **Heatmap**: Shows the similarity matrix.
- **Scatter Plot**: Projects embeddings into a 2D space using PCA.

---

## **How to Use the Application**

### **1. Single Text Analysis**
- Add **reference phrases** in the text area.
- Enter a single input text in the "Enter text to analyze" field.
- Click **Analyze** to get the top similar phrases along with similarity scores.

### **2. Multi-Text Analysis**
- Enter multiple texts (one per line).
- Click **Analyze Texts** to:
  - See a heatmap of similarity scores.
  - Visualize the embeddings in 2D space using PCA.

---

## **Testing the Application**

### **Sample Inputs**
#### **Reference Phrases**:
```
The weather is nice today
I love programming
Artificial intelligence is fascinating
ChatGPT is a powerful AI tool
Streamlit makes data visualization easy
```

#### **Single Text Analysis Input**:
```
AI is revolutionizing the world
```

#### **Multi-Text Analysis Inputs**:
```
Machine learning is a subset of AI
I enjoy building applications using Streamlit
Programming in Python is fun and efficient
OpenAI is a leader in generative AI
Visualization tools make data analysis better
```

### **Expected Results**
1. **Single Text Analysis**:
   - Displays the top 3 most similar reference phrases with similarity scores.

2. **Multi-Text Analysis**:
   - A **heatmap** showing pairwise similarity scores.
   - A **scatter plot** visualizing the relationship between text embeddings.

---

## **Technologies Used**

1. **Backend**:
   - **Azure OpenAI**: For generating text embeddings.
   - **Requests**: To interact with Azure APIs.

2. **Frontend**:
   - **Streamlit**: For the user interface.

3. **Visualization**:
   - **Plotly**: For interactive heatmaps and scatter plots.

4. **Machine Learning**:
   - **Scikit-learn**: For cosine similarity and PCA.

---

## **Future Enhancements**

1. **Batch Processing**:
   - Add support for larger datasets.

2. **Custom Models**:
   - Integrate fine-tuned models for specific use cases.

3. **Export Functionality**:
   - Allow users to download similarity matrices or scatter plots.

4. **Cloud Deployment**:
   - Host the app on services like AWS, GCP, or Streamlit Cloud.

---

## **GitHub Repository**

[Visit GitHub Repository](https://github.com/your-repo-name)

---
