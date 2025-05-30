# LLAMAREC: Cross-Domain Recommendations Through Large Language Model Intelligence
> Using Lightweight Language Models for Local, Cross-Domain Personalisation

LLAMAREC is a comprehensive research framework that leverages Large Language Models (LLMs) to perform cross-domain recommendations. The system addresses the challenge of recommending products across different categories by using the reasoning capabilities of LLMs to bridge preference patterns between domains like Books, Movies, Music, and more.

##  Key Features

- **Cross-Domain Intelligence**: Transfer user preferences between Books, Movies, Music, and TV shows
- **Local LLM Deployment**: Privacy-focused recommendations using locally-deployed Llama 3 models via Ollama
- **User Classification**: Sophisticated analysis of 'cold' users (<50 interactions) vs 'warm' users (>200 interactions)
- **Explainable Recommendations**: Human-readable explanations for every recommendation
- **Comprehensive Evaluation**: Rigorous metrics including Precision@K, Recall@K, NDCG, diversity, and novelty
- **Baseline Comparisons**: Neural Collaborative Filtering, content-based, and hybrid approaches
- **Statistical Analysis**: Advanced experimental designs with significance testing

## System Architecture

```
llamarec/
├── data/                           # Raw and processed datasets
│   ├── amazon_raw                  # Replace the dataset        
│   ├── processed                   # Using amazon_collector to .csv     
│   └── splits                      # Splits for cross domain, cold, warm, train and test 
├── models/                         # Machine learning models
│   └── neural_cf.py               # Neural Collaborative Filtering implementation
├── preprocessing/                  # Data processing pipeline
│   ├── amazon_collector.py        # Amazon dataset collection
│   ├── domain_preprocessor.py     # Cross-domain user identification
│   └── data_splitter.py          # Data splitting strategies
├── prompts/                       # Prompt engineering components
│   ├── prompt_generator.py        # Generate recommendation prompts
│   └── prompt_templates.py       # Template management
├── runners/                       # Execution scripts
│   ├── run_ollama_llamarec.py    # Main LLAMAREC runner
│   ├── run_cold_warm_scenarios.py # Cold/warm user analysis
│   └── evaluation_runner.py      # Automated evaluation
├── testers/                       # Interactive testing tools
│   ├── interactive_llamarec_tester.py  # Interactive demo interface
│   ├── run_enhanced_llamarec.py        # Enhanced testing capabilities
│   └── test_prompt_generator_cold_warm.py # Cold/warm specific testing
├── utils/                         # Utility functions
│   ├── llm_utils.py              # LLM interaction utilities
│   ├── file_utils.py             # File handling utilities
│   ├── performance_utils.py      # Performance optimisation
│   ├── logging_utils.py          # Structured logging
│   └── metrics.py                # Evaluation metrics
├── evaluations/                   # Evaluation framework
│   └── quantitative_evaluator.py # Comprehensive evaluation system
├── baselines/                     # Baseline comparison methods
│   ├── neural_cf.py              # NCF baseline implementation
│   └── ncf_evaluation_runner.py  # NCF evaluation runner
└── README.md
```

## Prerequisites

### System Requirements
- **Python 3.8+**
- **Memory**: 8GB+ RAM (16GB recommended for full datasets)
- **Storage**: ~100GB for processed datasets (Use can if you want https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **CPU**: Multi-core processor (parallel processing support)

### Required Software
- [Ollama](https://ollama.ai/) for local LLM deployment
- Git for version control
- I used ( https://ollama.com/library/llama3.1:8b )

##  Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llamarec.git
cd llamarec

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Ollama and Models

```bash
# Install Ollama (follow instructions at https://ollama.ai/)
# Pull required models
ollama pull llama3:8b
ollama pull llama3:70b  # Optional for comparison

# Start Ollama server
ollama serve
```

### 3. Data Preparation

```bash
# Download Amazon review datasets (Books, Movies & TV, CDs, Digital Music)
# Place JSON files in data/raw/

# Process Amazon datasets
python preprocessing/amazon_collector.py --base_path data/raw --output_dir data/processed

# Generate cross-domain user profiles
python preprocessing/domain_preprocessor.py --base_path data/processed --output_path data/splits

# Create data splits
python preprocessing/data_splitter.py --input_path data/splits --output_path data/splits
```

### 4. Generate Recommendations

```bash
# Interactive testing interface
python testers/interactive_llamarec_tester.py

# Run cold/warm user analysis
python runners/run_cold_warm_scenarios.py --mode comparative

# Basic LLAMAREC execution
python runners/run_ollama_llamarec.py
```

### 5. Evaluation

```bash
# Comprehensive evaluation
python runners/evaluation_runner.py

# Compare with baselines
python baselines/ncf_evaluation_runner.py --dataset data/splits/Books_filtered.csv \
    --user_history data/splits/user_history.json --domain Books
```

## Usage Examples

### Basic Cross-Domain Recommendation

```python
from testers.interactive_llamarec_tester import InteractiveLlamaRecTester

# Initialize tester
tester = InteractiveLlamaRecTester(
    user_history_path="data/splits/user_history.json",
    ollama_model="llama3:8b"
)

# Generate recommendations
recommendation = tester.generate_recommendation(
    user_id="A1B2C3D4E5",
    source_domain="Books",
    target_domain="Movies_and_TV",
    num_recs=3
)

print(recommendation['recommendations'])
```

### Cold vs Warm User Analysis

```python
from runners.run_cold_warm_scenarios import ColdWarmScenarioRunner
from testers.test_prompt_generator_cold_warm import PromptGeneratorTester

# Initialize components
base_tester = PromptGeneratorTester(splits_dir="data/splits")
scenario_runner = ColdWarmScenarioRunner(base_tester)

# Run comparative study
results = scenario_runner.run_comparative_study(
    max_users_per_scenario=20,
    temperature=0.7
)
```

### Comprehensive Evaluation

```python
from evaluations.quantitative_evaluator import RecommendationEvaluator

# Initialize evaluator
evaluator = RecommendationEvaluator("data/splits/user_history.json")

# Evaluate recommendations
results = evaluator.evaluate_recommendations(
    "results/llamarec_results.json",
    k=3
)

# Generate report
report = evaluator.generate_evaluation_report(
    results,
    "evaluation_report.txt"
)
```

##  Configuration

### Model Configuration

```python
# Configure different Llama models
model_configs = {
    "base_model": {
        "model_name": "llama3:8b",
        "temperature": 0.7,
        "max_tokens": 512
    },
    "large_model": {
        "model_name": "llama3:70b",
        "temperature": 0.7,
        "max_tokens": 512
    }
}
```

### Prompt Templates

```python
# Standard recommendation template
STANDARD_TEMPLATE = """
Based on the user's preferences in {source_domain}, generate personalised recommendations for items from {target_domain}.

User's highly-rated items: {user_history}
Target domain: {target_domain}

Recommend top 3 items with explanations.
"""

# Detailed reasoning template
DETAILED_TEMPLATE = """
You are an expert recommendation system. Analyse the user's preferences step-by-step.

Step 1: Analyse patterns in user's {source_domain} preferences: {user_history}
Step 2: Identify transferable preferences to {target_domain}
Step 3: Recommend 3 {target_domain} items that match these patterns

Provide detailed reasoning for each recommendation.
"""
```

##  Evaluation Metrics

### Ranking Quality
- **Precision@K**: Fraction of relevant items in top-K recommendations
- **Recall@K**: Fraction of relevant items retrieved from user's liked items
- **NDCG@K**: Normalised Discounted Cumulative Gain for ranking quality

### Recommendation Diversity
- **Intra-list Diversity**: Average pairwise dissimilarity within recommendations
- **Coverage**: Fraction of catalogue items recommended across all users
- **Novelty**: Dissimilarity from user's historical preferences

### System Performance
- **Response Time**: LLM inference latency
- **Success Rate**: Fraction of successful API calls
- **Quality Score**: Automated assessment of recommendation quality

##  Baseline Methods

### Neural Collaborative Filtering (NCF)
```bash
python baselines/ncf_evaluation_runner.py \
    --dataset data/splits/Books_filtered.csv \
    --user_history data/splits/user_history.json \
    --domain Books \
    --k 3
```

### Content-Based Filtering
```python
from baselines.content_based import ContentBasedRecommender

# Initialize with metadata
recommender = ContentBasedRecommender(metadata_df)
recommendations = recommender.recommend(user_liked_items, k=10)
```

##  Advanced Features

### Experimental Design
```python
from experiments.advanced_experimental_designs import AdvancedExperimentalDesigner

designer = AdvancedExperimentalDesigner(base_tester)

# Factorial experiment: Temperature × User Type × Domain × Template
factorial_results = designer.run_full_experiment_suite('factorial')

# Ablation study: Component contribution analysis
ablation_results = designer.run_full_experiment_suite('ablation')
```

### Statistical Analysis
```python
from experiments.advanced_experimental_designs import ExperimentalStatistics

stats = ExperimentalStatistics()

# Effect size calculation
effect_size = stats.compute_effect_size(group1_scores, group2_scores)
interpretation = stats.interpret_effect_size(effect_size)

# Power analysis
required_sample_size = stats.power_analysis(effect_size=0.5, power=0.8)
```

## Results and Findings

### Key Performance Insights
- **Cold Start Performance**: LLAMAREC demonstrates superior performance for users with limited interaction history
- **Cross-Domain Transfer**: Books↔Movies and Music↔Books show highest transfer success rates
- **Explainability**: 95%+ of recommendations include coherent explanations linking source preferences
- **User Type Differences**: Warm users receive 15-20% higher quality recommendations on average


## Contributing

We welcome contributions to LLAMAREC! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black llamarec/
flake8 llamarec/
```

## Citation

If you use LLAMAREC in your research, please cite our work:

```bibtex
@article{llamarec2025,
    title={LLAMAREC: Cross-Domain Recommendations Through Large Language Model Intelligence},
    author={Thanigai Singaravelan Senthil Kumar}
    supervised = {Dr Aboozar Taherkhani}
    year={2025},
    note={Available at: https://github.com/ThanigaiSingaravelan/llam2}
}
```


##  Acknowledgements

- **Ollama Team** for providing excellent local LLM deployment tools
- **Meta AI** for the Llama 3 model family
- **Amazon** for providing publicly available review datasets
- **De Montfort University** for research support and resources


##  Roadmap

### Upcoming Features
- [ ] Multi-modal recommendations (text + images)
- [ ] Real-time recommendation serving
- [ ] Advanced fine-tuning capabilities
- [ ] Extended domain support (Games, Electronics)
- [ ] Web-based user interface
- [ ] API endpoint for external integration

### Research Extensions
- [ ] Conversational recommendation interfaces
- [ ] Causal effect estimation in cross-domain transfer
- [ ] Long-term user satisfaction tracking
- [ ] Cultural and demographic bias analysis

---

**LLAMAREC** - Bridging domains through intelligent language model reasoning
