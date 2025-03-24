import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import networkx as nx
import numpy as np
import streamlit as st
from pypdf import PdfReader
from docx import Document

class TextSummarizer:
    def __init__(self):
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Error: Please install the spaCy model by running:")
            print("python -m spacy download en_core_web_sm")
            raise
    
    def summarize(self, text, percent=0.3):
        """
        Generate a summary of the given text using TextRank algorithm
        
        Args:
            text (str): The text to summarize
            percent (float): Percentage of original text to include in summary (0.1 to 0.5)
            
        Returns:
            str: The generated summary
        """
        if not isinstance(text, str) or text.strip() == "":
            return "No text provided for summarization."
            
        # Make sure percent is between 0.1 and 0.5
        percent = max(0.1, min(0.5, percent))
        
        return self.textrank_summary(text, percent)
    
    def textrank_summary(self, text, per):
        """Generate text summary using TextRank algorithm"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        if len(sentences) <= 1:
            return text
        
        # Create sentence vectors using spaCy's word vectors
        sentence_vectors = []
        for sent in sentences:
            # Skip sentences with no words with vectors
            if not any(token.has_vector for token in sent):
                sent_vec = np.zeros((len(sent), 96))  # Default embedding dimension
            else:
                words_with_vectors = [token.vector for token in sent if token.has_vector]
                if not words_with_vectors:
                    sent_vec = np.zeros(96)  # Default dimension
                else:
                    sent_vec = np.mean(words_with_vectors, axis=0)
            sentence_vectors.append(sent_vec)
        
        # Create similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])
        
        # Fill the similarity matrix
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    # Make sure we don't divide by zero
                    if np.linalg.norm(sentence_vectors[i]) * np.linalg.norm(sentence_vectors[j]) == 0:
                        sim_mat[i][j] = 0
                    else:
                        sim_mat[i][j] = self._cosine_similarity(sentence_vectors[i], sentence_vectors[j])
        
        # Create networkx graph and add edges with weights
        nx_graph = nx.from_numpy_array(sim_mat)
        
        # Apply PageRank algorithm
        scores = nx.pagerank(nx_graph)
        
        # Sort sentences by score and select top sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Calculate the number of sentences for the summary
        summary_size = max(1, int(len(sentences) * per))
        
        # Get top N sentences and sort them by original position
        top_sentences = sorted(ranked_sentences[:summary_size], key=lambda x: x[1])
        
        # Combine sentences into summary
        summary = " ".join([s.text for _, _, s in top_sentences])
        
        return summary
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0
        
        # Calculate cosine similarity
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file and return as a single string."""
    reader = PdfReader(pdf_file)
    text_content = []
    for page in reader.pages:
        text = page.extract_text()
        if text.strip():
            text_content.append(text.strip())
    return " ".join(text_content)

def extract_text_from_docx(docx_file):
    """Extract text from a Word document and return as a single string."""
    doc = Document(docx_file)
    text_content = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            text_content.append(paragraph.text.strip())
    return " ".join(text_content)

def extract_text(file):
    """Extract text from either PDF or Word document."""
    file_type = file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type in ['docx', 'doc']:
        return extract_text_from_docx(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

if __name__ == '__main__':
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None   
    if 'summary' not in st.session_state:
        st.session_state.summary = ""

    st.title("Text Summarizer")

    with st.sidebar:
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx', 'doc'])

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
        
        summary_percent = st.slider("Summary Length (%)", 
                                  min_value=10, 
                                  max_value=50, 
                                  value=30,
                                  help="Select the percentage of original text to keep in summary")
        
        if st.button("Generate Summary"):
            try:
                if st.session_state.uploaded_file:
                    with st.spinner("Extracting text and generating summary..."):
                        # Extract text from file
                        text = extract_text(uploaded_file)
                        if not text.strip():
                            st.error("Could not extract text from the file. Please make sure it's a text-based file.")
                            st.stop()
                        
                        # Initialize summarizer and generate summary
                        summarizer = TextSummarizer()
                        st.session_state.summary = summarizer.summarize(text, summary_percent/100)
                        st.success("Summary generated successfully!")
                else:
                    st.warning("Please upload a file first.")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    if st.session_state.summary:
        st.subheader("Generated Summary")
        st.markdown(f"{st.session_state.summary}")
        
        # Display original text length vs summary length
        original_words = len(text.split()) if 'text' in locals() else 0
        summary_words = len(st.session_state.summary.split())
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Text Length", f"{original_words} words")
        with col2:
            st.metric("Summary Length", f"{summary_words} words")
    
                






















