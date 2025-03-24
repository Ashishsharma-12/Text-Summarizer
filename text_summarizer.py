import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import networkx as nx
import numpy as np

class TextSummarizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Summarizer")
        self.root.geometry("800x600")
        self.root.configure(bg="#f5f5f5")
        
        # Initialize spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            messagebox.showerror("Model Error", "Please install the spaCy model by running:\npython -m spacy download en_core_web_sm")
            root.destroy()
            return
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#f5f5f5")
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            header_frame, 
            text="TextRank Summarizer", 
            font=("Arial", 18, "bold"),
            bg="#f5f5f5"
        ).pack(side=tk.LEFT)
        
        # Main content
        content_frame = tk.Frame(self.root, bg="#f5f5f5")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel for input
        left_frame = tk.LabelFrame(content_frame, text="Original Text", bg="#f5f5f5", font=("Arial", 10))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(left_frame, wrap=tk.WORD, font=("Arial", 11))
        self.input_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right panel for output
        right_frame = tk.LabelFrame(content_frame, text="Summary", bg="#f5f5f5", font=("Arial", 10))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, font=("Arial", 11))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controls panel
        controls_frame = tk.Frame(self.root, bg="#f5f5f5")
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Summary length selection
        tk.Label(
            controls_frame, 
            text="Summary length:", 
            bg="#f5f5f5", 
            font=("Arial", 10)
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        self.summary_percent = tk.StringVar(value="30%")
        summary_options = ["10%", "20%", "30%", "40%", "50%"]
        summary_dropdown = ttk.Combobox(
            controls_frame, 
            textvariable=self.summary_percent, 
            values=summary_options, 
            width=5,
            state="readonly"
        )
        summary_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        # Buttons
        self.summarize_btn = tk.Button(
            controls_frame, 
            text="Summarize", 
            command=self.summarize_text,
            bg="#4CAF50", 
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.summarize_btn.pack(side=tk.LEFT, padx=5)
        
        self.clear_btn = tk.Button(
            controls_frame, 
            text="Clear", 
            command=self.clear_fields,
            bg="#f44336", 
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            anchor=tk.W, 
            bg="#e0e0e0", 
            relief=tk.SUNKEN,
            padx=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def summarize_text(self):
        """Generate a summary of the input text using TextRank"""
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Warning", "Please enter text to summarize.")
            return
        
        # Update status
        self.status_var.set("Summarizing...")
        self.root.update_idletasks()
        
        try:
            # Parse percentage
            percent = int(self.summary_percent.get().strip('%'))
            summary = self.textrank_summary(text, percent/100)
            
            # Display summary
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, summary)
            
            # Update status
            self.status_var.set(f"Summary complete. ({len(summary.split())} words)")
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred during summarization.")
    
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
    
    def clear_fields(self):
        """Clear input and output text fields"""
        self.input_text.delete("1.0", tk.END)
        self.output_text.delete("1.0", tk.END)
        self.status_var.set("Ready")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = TextSummarizerApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Application error: {str(e)}") 