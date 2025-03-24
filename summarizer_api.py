import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
import networkx as nx
import numpy as np

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


# Example usage
if __name__ == "__main__":
    # Sample text for demonstration
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.
    The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". 
    This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    AI applications include advanced web search engines, recommendation systems, understanding human speech, self-driving cars, automated decision-making and competing at the highest level in strategic game systems.
    As machines become increasingly capable, tasks considered to require "intelligence" are often removed from the definition of AI, a phenomenon known as the AI effect. 
    For instance, optical character recognition is frequently excluded from things considered to be AI, having become a routine technology.
    """
    
    # Create a summarizer
    summarizer = TextSummarizer()
    
    # Generate summary at 30% length
    summary = summarizer.summarize(sample_text, 0.5)
    
    # Print results
    print("Original Text Length:", len(sample_text.split()), "words")
    print("Summary Length:", len(summary.split()), "words")
    print("\n--- SUMMARY ---\n")
    print(summary)

