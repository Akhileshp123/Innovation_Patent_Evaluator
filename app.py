
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import sys
import re

# --- Configuration ---
# NOTE: The 'all-MiniLM-L6-v2' model will be downloaded locally by sentence-transformers
# upon the first execution. This is the necessary open-source AI component.
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
MOCK_FILE_PATH = 'mock_patent_corpus.csv'
# Set Matplotlib backend to Agg for non-interactive environments (like a terminal/server)
plt.switch_backend('Agg')

# --- 1. Data Generation and Loading ---

def generate_mock_corpus(filepath=MOCK_FILE_PATH, num_patents=50):
    """
    Generates a mock dataset of existing patents for the patent corpus.
    This replaces a real database/API to satisfy the 'no static data'
    and 'no API' constraints for demonstration purposes.
    """
    print(f"Generating a mock patent corpus of {num_patents} patents...")

    titles = [
        "AI-driven dynamic energy allocation system",
        "Blockchain authenticated supply chain ledger",
        "Quantum computing error correction method",
        "Self-adjusting robotic surgical instrument",
        "Biometric security system using retinal scanning",
        "Haptic feedback virtual reality glove",
        "Advanced drone navigation via atmospheric monitoring",
        "Solar panel efficiency booster using graphene oxide",
        "Decentralized autonomous organization (DAO) framework",
        "Adaptive cooling technology for server farms",
        "Personalized digital twin for healthcare monitoring",
        "Electrostatic dust removal system for aerospace",
        "Automated emotional recognition via voice analysis",
        "High-density solid-state battery architecture",
        "Predictive maintenance for wind turbines using edge AI"
    ]

    corpus_data = []
    start_year = 2018
    end_year = 2024

    for i in range(num_patents):
        # Select a random year and title
        year = np.random.randint(start_year, end_year + 1)
        title_base = np.random.choice(titles)
        
        # Create a mock description
        description = f"Patent {i+1} details an invention related to {title_base.lower().replace('-', ' ')}. "
        description += "The core claims involve a unique combination of [Technology A], [Technology B], and [Innovation Metric]."
        
        # Assign a random CPC class (simulating patent classification)
        cpc_class = np.random.choice(['G06F', 'H04L', 'A61B', 'F03D', 'B60R'])

        corpus_data.append({
            'Patent_ID': f'US{year}0{i+1:04d}',
            'Title': title_base + f" ({i+1})",
            'Abstract': description,
            'CPC_Class': cpc_class,
            'Filing_Year': year,
            'Citations': np.random.randint(0, 15)
        })

    df = pd.DataFrame(corpus_data)
    
    # Save for potential inspection/reuse
    df.to_csv(filepath, index=False)
    print(f"Mock corpus saved to {filepath}. Using in-memory DataFrame for analysis.")
    return df

def load_input_patent(filepath):
    """Loads the new patent text from a file (e.g., input_patent.txt)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Simple extraction logic for demonstration
        title_match = re.search(r'Title:\s*(.*)', content, re.IGNORECASE)
        abstract_match = re.search(r'Abstract:\s*([\s\S]*?)(?:\n\n|\Z)', content, re.IGNORECASE)
        
        input_title = title_match.group(1).strip() if title_match else "Untitled New Invention"
        input_abstract = abstract_match.group(1).strip() if abstract_match else content.strip()
        
        if not input_abstract:
             raise ValueError("Input file is empty or missing patent content.")

        return input_title, input_abstract
    
    except FileNotFoundError:
        print(f"\nERROR: Input patent file not found at '{filepath}'.")
        print("Please ensure the path is correct and the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred loading the input file: {e}")
        sys.exit(1)

# --- 2. LLM Simulation for Novelty and Innovation Assessment ---

def llm_novelty_assessment(patent_text):
    """
    SIMULATED LLM Analysis for Novelty and Innovation.
    
    In a real-world scenario, a large model (like a local Llama/Mistral) would
    parse the text for key features, identify gaps in prior art, and provide
    a detailed, multi-factor analysis.
    
    Here, we simulate this by using simple NLP rules and heuristics.
    """
    print("\n--- Running Simulated LLM Novelty Assessment ---")
    
    # Heuristics based on text complexity and uniqueness of certain terms
    complexity_score = len(patent_text.split()) / 150 # Score based on length
    unique_words = len(set(re.findall(r'\b\w+\b', patent_text.lower())))
    
    # Check for keywords indicating innovation/novelty
    novelty_keywords = ['quantum', 'decentralized', 'graphene', 'biometric', 'ai-driven', 'personalized digital twin']
    keyword_count = sum(1 for word in novelty_keywords if word in patent_text.lower())
    
    # Calculate initial raw score
    raw_score = (complexity_score * 0.4) + (keyword_count * 0.6)
    novelty_score = min(100, max(40, raw_score * 15)) # Scale to 40-100
    
    # Generate qualitative assessment based on the score
    if novelty_score > 85:
        assessment = ("High Innovation Potential. The claims utilize a sophisticated combination of emerging technologies. "
                      "The technical language suggests a significant step beyond current prior art. The market disruption potential is substantial.")
    elif novelty_score > 65:
        assessment = ("Moderate Innovation. The core idea is sound, building effectively on established principles but with "
                      "a clever, non-obvious combination of features. Further analysis of competitive patents is crucial.")
    else:
        assessment = ("Lower Novelty Risk. The description, while clear, appears closely aligned with existing, well-cited "
                      "technologies. The scope of claims may be narrow, requiring a deeper search for similar prior art.")
    
    return novelty_score, assessment

# --- 3. Similarity Scoring (Embedding Model) ---

def load_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    """Loads the sentence-transformer model."""
    try:
        print(f"Loading open-source embedding model: {model_name}...")
        model = SentenceTransformer(model_name)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"\nERROR: Could not load the embedding model '{model_name}'.")
        print("Please ensure your environment has network access for initial download, or that the model is cached.")
        print(f"Details: {e}")
        sys.exit(1)

def score_similarity(model, new_patent_text, corpus_df):
    """Calculates cosine similarity between the new patent and the corpus."""
    
    corpus_texts = corpus_df['Abstract'].tolist()
    all_texts = [new_patent_text] + corpus_texts
    
    # Encode all texts
    embeddings = model.encode(all_texts, convert_to_tensor=False, show_progress_bar=False)
    
    # New patent is the first embedding
    new_patent_embedding = embeddings[0].reshape(1, -1)
    corpus_embeddings = embeddings[1:]
    
    # Calculate Cosine Similarity
    similarities = cosine_similarity(new_patent_embedding, corpus_embeddings)[0]
    
    corpus_df['Similarity_Score'] = similarities
    
    # Sort and get top N similar patents
    top_n = 5
    most_similar = corpus_df.sort_values(by='Similarity_Score', ascending=False).head(top_n).copy()
    
    # Determine the "Novelty Gap" based on the highest similarity
    highest_score = most_similar['Similarity_Score'].iloc[0]
    novelty_gap = 1.0 - highest_score # 1.0 means identical, 0.0 means completely different
    
    return most_similar, novelty_gap

# --- 4. Patent Landscape and Trend Analysis ---

def perform_trend_analysis(corpus_df):
    """
    Performs time-series analysis on the mock patent corpus to identify trends
    in the "patent population" (corpus distribution).
    """
    
    # 1. Patent Population by Year
    yearly_counts = corpus_df.groupby('Filing_Year').size().reset_index(name='Count')
    
    # 2. Patent Population by Class (Top 5)
    class_counts = corpus_df['CPC_Class'].value_counts().nlargest(5).reset_index()
    class_counts.columns = ['CPC_Class', 'Count']
    
    # 3. Citation Analysis (Innovation Metric)
    citation_trends = corpus_df.groupby('Filing_Year')['Citations'].mean().reset_index(name='Avg_Citations')

    return yearly_counts, class_counts, citation_trends

# --- 5. Visualization ---

def create_visualizations(yearly_counts, class_counts, citation_trends, report_title="Patent Analysis Visualizations"):
    """Generates and saves the required trend visualizations."""
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    fig.suptitle(report_title, fontsize=16, y=1.02)
    
    # --- Plot 1: Trend Visualization (Yearly Filings) ---
    axes[0].plot(yearly_counts['Filing_Year'], yearly_counts['Count'], marker='o', linestyle='-', color='#3b82f6')
    axes[0].set_title('Trend Analysis: Patent Population by Filing Year (2018-2024)', fontsize=12)
    axes[0].set_xlabel('Filing Year')
    axes[0].set_ylabel('Number of Patents Filed')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].ticklabel_format(style='plain', axis='x')
    
    # --- Plot 2: Population Analysis (Top CPC Classes) ---
    axes[1].bar(class_counts['CPC_Class'], class_counts['Count'], color=['#10b981', '#f59e0b', '#ef4444', '#6366f1', '#ec4899'])
    axes[1].set_title('Population Analysis: Distribution by Top Patent Classification (CPC)', fontsize=12)
    axes[1].set_xlabel('CPC Class')
    axes[1].set_ylabel('Total Count')
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # --- Plot 3: Innovation Metric Trend (Average Citations) ---
    axes[2].plot(citation_trends['Filing_Year'], citation_trends['Avg_Citations'], marker='s', linestyle='--', color='#2563eb')
    axes[2].set_title('Innovation Metric: Average Citations Over Time', fontsize=12)
    axes[2].set_xlabel('Filing Year')
    axes[2].set_ylabel('Average Citations')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].ticklabel_format(style='plain', axis='x')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    visualization_path = 'patent_trends_visualization.png'
    plt.savefig(visualization_path)
    plt.close() # Close the figure to free up memory
    print(f"\nVisualizations saved to '{visualization_path}'")
    return visualization_path

# --- 6. Report Generation ---

def generate_report(input_title, novelty_score, llm_assessment, most_similar, novelty_gap, visualization_path, input_file_path):
    """Generates the final structured report based on all analyses."""

    report = f"""
# Innovation Patent AI Evaluation Report

## A. Input Patent Details
- **Title:** {input_title}
- **Input Source File Path:** {input_file_path}
- **Evaluation Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## B. Novelty and Innovation Assessment (Simulated LLM)

### Novelty Score
- **Overall Score (0-100):** **{novelty_score:.1f}**

### Conservation Analysis (Qualitative Assessment)
*This section provides a high-level, human-readable summary of the invention's potential, simulating the output of a large-scale LLM analysis.*

> {llm_assessment}

---

## C. Similarity and Prior Art Analysis

### Novelty Gap
- **Novelty Gap Score (1.0 = Highly Novel):** **{novelty_gap:.3f}**
- **Interpretation:** This score represents how far the invention is from the most similar existing patent in the corpus. A score closer to $1.0$ indicates greater novelty and a lower risk of obviousness.

### Most Similar Patents (Prior Art)
*The following patents from the corpus exhibit the highest semantic similarity to the input invention, calculated using Cosine Similarity on Sentence Embeddings.*

| Similarity Score | Patent ID | Title |
|:----------------:|:---------:|:------|
"""
    for index, row in most_similar.iterrows():
        score = f"{row['Similarity_Score']:.4f}"
        report += f"| {score} | {row['Patent_ID']} | {row['Title']} |\n"

    report += "\n---\n"
    
    # Protection Recommendations (Mapping the required output)
    recommendation = ""
    if novelty_gap > 0.70:
        recommendation = ("**Broad Scope:** The novelty gap is significant. We recommend pursuing claims with broad language to cover "
                          "multiple embodiments and downstream applications. Focus on protecting the core functional method.")
    elif novelty_gap > 0.40:
        recommendation = ("**Intermediate Scope:** Novelty exists in the combination of features. Claims should focus on the "
                          "unique integration points identified in the abstract and ensure that all existing similar patents "
                          "are explicitly differentiated in the final claim language.")
    else:
        recommendation = ("**Narrow Scope:** The invention shows high similarity to existing prior art. Claims should be highly "
                          "specific, focusing on small, non-obvious improvements or parameter limitations (e.g., specific "
                          "materials, tolerances, or data processing steps).")

    report += f"""
## D. Protection Recommendations

### Recommended Scope Strategy
{recommendation}

---

## E. Population Analysis Trends Visualization

*The following image provides an overview of the patent landscape (population) based on the current corpus data, illustrating the time-series trends requested.*

![Patent Trends Visualization]({visualization_path})

"""
    return report

# --- Main Execution ---

if __name__ == '__main__':
    # 1. Determine Input File Path
    default_input_path = 'input_patent.txt'
    if len(sys.argv) > 1:
        input_file_to_use = sys.argv[1]
        print(f"Using input patent file from command line argument: '{input_file_to_use}'")
        print("-" * 50)
    else:
        input_file_to_use = default_input_path
        print(f"No input file path provided. Checking for default file: '{input_file_to_use}'")
        
        # If no argument is provided and the default file doesn't exist, create a placeholder.
        if not os.path.exists(input_file_to_use):
            print(f"Creating a placeholder input file: '{input_file_to_use}'")
            with open(input_file_to_use, 'w') as f:
                f.write("Title: Novel Graphene-Based Quantum Battery Architecture\n\n")
                f.write("Abstract: A high-density solid-state battery architecture utilizing a synthesized three-dimensional graphene oxide "
                        "lattice structure to facilitate ultra-rapid charge transfer and increase energy retention by 40% compared to "
                        "traditional lithium-ion cells. The core innovation lies in the quantum tunneling effects managed by the specific "
                        "lattice geometry, enabling unprecedented power output for mobile devices and electric vehicles.")
            print(f"**NOTE**: Please edit '{input_file_to_use}' with your specific patent claims/abstract for a real analysis.")
            print("-" * 50)
    
    # 2. Load/Generate Data
    input_title, input_abstract = load_input_patent(input_file_to_use)
    patent_corpus_df = generate_mock_corpus()
    
    # 3. LLM Novelty Assessment
    novelty_score, llm_assessment = llm_novelty_assessment(input_abstract)
    
    # 4. Similarity Scoring
    embedding_model = load_embedding_model()
    most_similar_patents, novelty_gap = score_similarity(embedding_model, input_abstract, patent_corpus_df)
    
    # 5. Trend Analysis
    yearly_counts, class_counts, citation_trends = perform_trend_analysis(patent_corpus_df)
    
    # 6. Visualization
    visualization_path = create_visualizations(yearly_counts, class_counts, citation_trends, f"Evaluation for: {input_title}")
    
    # 7. Report Generation
    final_report = generate_report(
        input_title, 
        novelty_score, 
        llm_assessment, 
        most_similar_patents, 
        novelty_gap, 
        visualization_path,
        input_file_to_use # Pass the file path to the report generator
    )
    
    print("\n" + "=" * 80)
    print(f"Report Generated for: {input_title}")
    print(f"Novelty Score: {novelty_score:.1f}")
    print(f"Novelty Gap: {novelty_gap:.3f}")
    print("=" * 80 + "\n")
    
    # Save the report to a markdown file
    report_filepath = "patent_evaluation_report.md"
    with open(report_filepath, 'w', encoding='utf-8') as f:
        f.write(final_report)
    print(f"FULL REPORT SAVED TO: '{report_filepath}'")
