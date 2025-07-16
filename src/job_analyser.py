"""
Job Analyzer Utility with External Prompt Files

A utility class for analyzing job postings using LLMs to extract structured information
for automated cover letter and resume generation.
"""

import json
import time
import requests
from typing import Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel


class JobAnalysis(BaseModel):
    """Structured output from job posting analysis"""
    company_name: str
    job_title: str
    context: str
    research_topics: List[str]
    required_skills: List[str]
    preferred_skills: List[str]
    key_responsibilities: List[str]
    company_values: List[str]
    keywords_to_emphasize: List[str]
    experience_level: str
    location: Optional[str] = None
    salary_range: Optional[str] = None

    def save_to_file(self, filepath: str) -> None:
        """Save analysis to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
        print(f"Analysis saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: str) -> 'JobAnalysis':
        """Load analysis from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


class JobAnalyzer:
    """
    Analyzes job postings using LLM to extract structured information
    for targeted application generation.
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434", 
                 model: str = "mistral:7b-instruct-q4_0",
                 use_quantized: bool = True,
                 verbose: bool = True,
                 prompt_file: str = "prompts/job_analyser/main_prompt.txt"):
        """
        Initialize JobAnalyzer
        
        Args:
            ollama_url: Ollama API endpoint
            model: Model name to use
            use_quantized: Whether to use quantized models
            verbose: Whether to print diagnostics
            prompt_file: Path to the prompt template file
        """
        self.ollama_url = ollama_url
        self.use_quantized = use_quantized
        self.verbose = verbose
        self.prompt_file = prompt_file
        
        # Load prompt template
        self._load_prompt_template()
        
        # Model selection based on quantization preference
        if use_quantized:
            # Ollama quantized models (much smaller, faster)
            model_options = {
                "llama3.1": "llama3.1:8b-instruct-q4_0",  # ~4.6GB
                "llama3.2": "llama3.2:3b-instruct-q4_K_M", # ~2GB  
                "mistral": "mistral:7b-instruct-q4_0",     # ~4.1GB
                "codellama": "codellama:7b-instruct-q4_0", # ~4.1GB
            }
            self.model = model if ":" in model else model_options.get(model, "llama3.2:3b-instruct-q4_K_M")
        else:
            # Full precision models (larger, slower)
            self.model = model if ":" in model else f"{model}:latest"
        
        if self.verbose:
            print(f"JobAnalyzer initialized")
            print(f"   Model: {self.model}")
            print(f"   Quantized: {use_quantized}")
            print(f"   Prompt file: {self.prompt_file}")
        
        self._check_model_availability()

    def _load_prompt_template(self) -> None:
        """Load prompt template from file"""
        try:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            if self.verbose:
                print(f"Loaded prompt template from {self.prompt_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        except Exception as e:
            raise Exception(f"Error loading prompt template: {e}")

    def _check_model_availability(self) -> None:
        """Check if the model is available, if not suggest how to get it"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                if self.model not in available_models:
                    print(f"Model '{self.model}' not found locally.")
                    print(f"To download it, run: ollama pull {self.model}")
                    if available_models:
                        print(f"Available models: {available_models}")
                elif self.verbose:
                    print(f"Model '{self.model}' is available")
            else:
                print("Could not check model availability")
        except requests.exceptions.RequestException:
            print("Could not connect to Ollama. Make sure it's running: ollama serve")

    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama with timing and token diagnostics"""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                }
            )
            response.raise_for_status()
            
            end_time = time.time()
            response_data = response.json()
            
            # Print diagnostics if verbose
            if self.verbose:
                print(f"LLM Diagnostics:")
                print(f"   Model: {self.model}")
                print(f"   Generation time: {end_time - start_time:.2f}s")
                print(f"   Prompt tokens: {response_data.get('prompt_eval_count', 'N/A')}")
                print(f"   Response tokens: {response_data.get('eval_count', 'N/A')}")
                if response_data.get('eval_count'):
                    print(f"   Tokens/sec: {response_data.get('eval_count', 0) / (end_time - start_time):.1f}")
                print(f"   Response length: {len(response_data['response'])} chars")
                print("-" * 50)
            
            return response_data["response"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Ollama: {e}")

    def _create_analysis_prompt(self, job_text: str) -> str:
        """Create the analysis prompt using the loaded template"""
        return self.prompt_template.format(job_text=job_text)

    def analyze_job_posting(self, job_text: str) -> JobAnalysis:
        """
        Analyze job posting and extract structured information
        
        Args:
            job_text: Raw job posting text
            
        Returns:
            JobAnalysis object with structured data
            
        Raises:
            Exception: If analysis fails
        """
        if self.verbose:
            print(f"Analyzing job posting ({len(job_text)} characters)...")
        
        prompt = self._create_analysis_prompt(job_text)
        
        try:
            response = self._call_ollama(prompt)
            
            # Parse JSON response
            analysis_data = json.loads(response)
            
            # Validate and create JobAnalysis object
            analysis = JobAnalysis(**analysis_data)
            
            if self.verbose:
                print(f"Analysis completed successfully")
                
            return analysis
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise Exception(f"Error analyzing job posting: {e}")

    def analyze_from_file(self, filepath: str) -> JobAnalysis:
        """
        Analyze job posting from text file
        
        Args:
            filepath: Path to job posting text file
            
        Returns:
            JobAnalysis object
        """
        file_path = Path(filepath)
        if not file_path.exists():
            raise FileNotFoundError(f"Job posting file not found: {filepath}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            job_text = f.read().strip()
        
        if self.verbose:
            print(f"Reading job posting from: {filepath}")
        
        return self.analyze_job_posting(job_text)

    def print_analysis(self, analysis: JobAnalysis) -> None:
        """Pretty print the analysis results"""
        print(f"\n{'='*60}")
        print(f"JOB ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Company: {analysis.company_name}")
        print(f"Role: {analysis.job_title}")
        print(f"Experience Level: {analysis.experience_level}")
        if analysis.location:
            print(f"Location: {analysis.location}")
        if analysis.salary_range:
            print(f"Salary: {analysis.salary_range}")
        
        print(f"\nContext:")
        print(f"   {analysis.context}")
        
        print(f"\nResearch Topics ({len(analysis.research_topics)}):")
        if analysis.research_topics:
            for topic in analysis.research_topics:
                print(f"   • {topic}")
        else:
            print("   None identified")
            
        print(f"\nRequired Skills ({len(analysis.required_skills)}):")
        for skill in analysis.required_skills:
            print(f"   • {skill}")
        
        print(f"\nPreferred Skills ({len(analysis.preferred_skills)}):")
        for skill in analysis.preferred_skills:
            print(f"   • {skill}")
            
        print(f"\nKey Responsibilities ({len(analysis.key_responsibilities)}):")
        for resp in analysis.key_responsibilities:
            print(f"   • {resp}")
            
        print(f"\nCompany Values ({len(analysis.company_values)}):")
        for value in analysis.company_values:
            print(f"   • {value}")
            
        print(f"\nKeywords to Emphasize ({len(analysis.keywords_to_emphasize)}):")
        for keyword in analysis.keywords_to_emphasize:
            print(f"   • {keyword}")


# Example usage and testing
if __name__ == "__main__":
    # Test the utility
    sample_job = """
    Senior AI Researcher - Aleph Alpha Research
    
    We are building category-defining AI innovation for European AI sovereignty.
    Focus on multi-modal transformer architectures and explainable deep learning.
    
    Required: PhD in CS, 3+ years PyTorch, self-supervised learning experience
    Preferred: Experience with 10B+ parameter models, scientific publications
    
    You'll develop novel approaches for model interpretability and work end-to-end
    from research to production deployment.
    """
    
    print("Testing JobAnalyzer with External Prompts")
    print("=" * 50)
    
    try:
        # Initialize analyzer with custom prompt file
        analyzer = JobAnalyzer(
            verbose=True,
            prompt_file="prompts/job_analyser/main_prompt.txt"
        )
        
        # Analyze job posting
        analysis = analyzer.analyze_job_posting(sample_job)
        
        # Print results
        analyzer.print_analysis(analysis)
        
        # Save to file
        analysis.save_to_file("test_analysis.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the prompt file exists at the specified path")
        print("2. Make sure Ollama is running: ollama serve")
        print("3. Install a model: ollama pull llama3.2:3b-instruct-q4_K_M")