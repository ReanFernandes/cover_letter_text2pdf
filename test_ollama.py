import json
import requests
from typing import Dict, List, Optional
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

class JobAnalyzer:
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434", 
                 model: str = "llama3.1:8b-instruct-q4_0",
                 use_quantized: bool = True):
        self.ollama_url = ollama_url
        self.use_quantized = use_quantized
        
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
        
        print(f"Using model: {self.model}")
        print(f"Quantized: {use_quantized}")
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check if the model is available, if not suggest how to get it"""
        try:
            # List available models
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                available_models = [model["name"] for model in response.json()["models"]]
                if self.model not in available_models:
                    print(f"⚠️  Model '{self.model}' not found locally.")
                    print(f"📥 To download it, run: ollama pull {self.model}")
                    print(f"Available models: {available_models}")
                else:
                    print(f"✅ Model '{self.model}' is available")
            else:
                print("⚠️  Could not check model availability")
        except requests.exceptions.RequestException:
            print("⚠️  Could not connect to Ollama. Make sure it's running: ollama serve")

    def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama with timing and token diagnostics"""
        import time
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"  # Request JSON format
                }
            )
            response.raise_for_status()
            
            end_time = time.time()
            response_data = response.json()
            
            # Print diagnostics
            print(f"🔍 LLM Diagnostics:")
            print(f"   Model: {self.model}")
            print(f"   Generation time: {end_time - start_time:.2f}s")
            print(f"   Prompt tokens: {response_data.get('prompt_eval_count', 'N/A')}")
            print(f"   Response tokens: {response_data.get('eval_count', 'N/A')}")
            print(f"   Tokens/sec: {response_data.get('eval_count', 0) / (end_time - start_time):.1f}")
            print(f"   Response length: {len(response_data['response'])} chars")
            print("-" * 50)
            
            return response_data["response"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Ollama: {e}")
    
    def analyze_job_posting(self, job_text: str) -> JobAnalysis:
        """Analyze job posting and extract structured information"""
        
        prompt = f"""
        You are an expert job posting analyzer. Analyze the following job posting and extract key information.

        IMPORTANT: Return ONLY valid JSON in the exact format specified below. Do not include any other text.

        Required JSON format:
        {{
            "company_name": "string",
            "job_title": "string",
            "context": "string",
            "research_topics": ["topic1", "topic2"] # these will be topics that are about the companies products which warrant further research,
            "required_skills": ["skill1", "skill2"],
            "preferred_skills": ["skill1", "skill2"],
            "key_responsibilities": ["responsibility1", "responsibility2"],
            "company_values": ["value1", "value2"],
            "keywords_to_emphasize": ["keyword1", "keyword2"],
            "experience_level": "entry/mid/senior/lead",
            "location": "string or null",
            "salary_range": "string or null"
        }}

        Guidelines:
        - context: 1-2 sentences about what makes this company/role uniquely interesting or different. Focus on their mission, unique technology, or market position - NOT generic job responsibilities.
        - research_topics: ONLY include specific technical research areas, proprietary methodologies, or company-specific initiatives that are central to their work. Do NOT include employee benefits, generic technical terms, or common job requirements.
        - Extract 5-10 most important technical skills for each category
        - Keywords to emphasize should be domain-specific terms that appear multiple times or seem central to the role
        - Company values should be explicitly stated principles or cultural values from the posting
        - Be specific with technical skills (e.g., "PyTorch" not just "deep learning framework")
        - For experience level, choose the best fit from: entry, mid, senior, lead
        - If no specific research topics are mentioned, return an empty array

        Job Posting:
        {job_text}
        """
        
        try:
            response = self._call_ollama(prompt)
            
            # Parse JSON response
            analysis_data = json.loads(response)
            
            # Validate and create JobAnalysis object
            return JobAnalysis(**analysis_data)
            
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse LLM response as JSON: {e}")
        except Exception as e:
            raise Exception(f"Error analyzing job posting: {e}")

    def print_analysis(self, analysis: JobAnalysis) -> None:
        """Pretty print the analysis results"""
        print(f"\n{'='*50}")
        print(f"JOB ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Company: {analysis.company_name}")
        print(f"Role: {analysis.job_title}")
        print(f"Experience Level: {analysis.experience_level}")
        print(f"Context: {analysis.context}")
        
        print(f"\nResearch Topics ({len(analysis.research_topics)}):")
        if analysis.research_topics:
            for topic in analysis.research_topics:
                print(f"  🔍 {topic}")
        else:
            print("  None identified")
            
        print(f"\nRequired Skills ({len(analysis.required_skills)}):")
        for skill in analysis.required_skills:
            print(f"  • {skill}")
        
        print(f"\nPreferred Skills ({len(analysis.preferred_skills)}):")
        for skill in analysis.preferred_skills:
            print(f"  • {skill}")
            
        print(f"\nKey Responsibilities ({len(analysis.key_responsibilities)}):")
        for resp in analysis.key_responsibilities:
            print(f"  • {resp}")
            
        print(f"\nCompany Values ({len(analysis.company_values)}):")
        for value in analysis.company_values:
            print(f"  • {value}")
            
        print(f"\nKeywords to Emphasize ({len(analysis.keywords_to_emphasize)}):")
        for keyword in analysis.keywords_to_emphasize:
            print(f"  • {keyword}")

# Example usage with different quantization options
if __name__ == "__main__":
    # Test with a sample job posting
    sample_job = """Stellenbeschreibung

Aus 1 mach 91.000 – so beginnt unsere Unternehmensgeschichte. Und sie ist noch lange nicht zu Ende erzählt: Inhabergeführt, in den vergangenen Jahrzehnten kontinuierlich gewachsen und heute einer der größten unabhängigen IT- und Businessdienstleister der Welt. Durch unsere Kundennähe entstehen vertrauensvolle Beziehungen und unsere Branchen- und Technologiekompetenz ermöglicht es, die Bedürfnisse Ihrer Zielgruppen zu erfüllen.

Dein Herz schlägt für Daten und Machine-Learning? Du hast vielleicht sogar erste Erfahrungen mit Projekten im Data Science oder KI-Umfeld gesammelt? Du suchst nach einer sinnstiftenden Tätigkeit und der Chance, einen echten Mehrwert für die Gesellschaft zu schaffen? Dann werde Teil unserer Erfolgsgeschichte und gestalte mit uns in anspruchsvollen Projekten die digitale Zukunft Deutschlands!

Als AI Developer:in unterstützt Du ein hochdynamisches Team bei der Implementierung anspruchsvoller KI-Projekte bei Kunden im öffentlichen Dienst. Du hilfst uns bei der Umsetzung innovativer Projekte, insbesondere in der Verwendung von Large Language Models und anderen Foundation Models zur Steigerung der Effizienz der Verwaltungsarbeit. Gemeinsam schaffen wir die Verwaltung des 21. Jahrhunderts, zum Beispiel durch die Bereitstellung digitaler Assistenten oder bei der Digitalisierung von Verwaltungsvorgängen.

Aufgaben

 Du identifizierst Muster, Zusammenhänge und Ansätze für Prozessoptimierungen.
 Du übernimmst Management und Governance von Trainingsdaten
 Du bist zuständig für eine verantwortungsvolle und ethische KI-Entwicklung, insbesondere unter Einhaltung der Vorgaben des AI Acts
 Für unsere Kunden konzipierst und entwickelst du KI-Lösungen, die Data Engineering, Modellentwicklung und Bereitstellung der Modelle umfassen.
 Du arbeitest mit verschiedenen IT- und Fachbereichen zusammen und befindest dich in einem anspruchsvollen und abwechslungsreichen Projektumfeld.

Qualifikation

 Du hast du ein Studium im Bereich (Wirtschafts-) Informatik, Mathematik, Statistik oder eine vergleichbare Ausbildung absolviert.
 Du kennst dich gut mit den verschiedenen Machine Learning Verfahren und Modellarten aus.
 Du hast idealerweise bereits zwei oder mehr Jahre Berufserfahrung im Bereich Data Science oder Anwendungen von KI.
 Du bist mit dem einschlägigen Software-Stack vertraut und hast damit bereits Erfahrungen in Projekten gesammelt.
 Du hast Erfahrungen in der Aufbereitung und Analyse von Daten sowie in der Konzeption und Implementierung von Daten-Pipelines.
 Erfahrungen mit Natural Language Processing, insbesondere im Umgang mit großen Sprachmodellen sind von großem Vorteil
 Du löst mit Begeisterung komplexe Aufgaben und eine bereichsübergreifende Denkweise zeichnet dich aus.
 Teamfähigkeit, Zuverlässigkeit und Engagement zählen zu deinen Stärken.
 Deutsch und Englisch beherrschst du fließend in Wort und Schrift (Deutsch mindestens auf B2-Niveau).

WAS WIR BIETEN

 Bei uns findest du Kolleg:innen, mit denen die Zusammenarbeit Spaß macht. Wir begegnen uns offen, duzen uns über alle Positionen hinweg und denken nicht in Hierarchien oder Silos.
 Du arbeitest meist direkt an deinem Heimatort – weil wir Kundennähe wörtlich nehmen und uns Work-Life-Balance am Herzen liegt.
 Du profitierst von flexiblen Arbeitszeiten und hast je nach Kundensituation die Möglichkeit, von zuhause zu arbeiten.
 Die richtigen Trainings und Zertifikate bringen deine Weiterbildung voran. Unsere E-Learning-Plattform Academia ermöglicht dir das Lernen, wo und wann du willst.
 Einen Teil deines Bruttogehalts kannst du in CGI Aktien investieren – bis maximal 3 % des Monatsgehalts geben wir für jeden Euro einen weiteren hinzu.
 Außerdem beteiligen wir dich am Unternehmenserfolg: Du erhältst eine Gewinnbeteiligung, die sich nach deiner individuellen Leistung richtet, sowie danach, wie wir als Unternehmen unsere finanziellen Ziele erreichen konnten.
 Wir bieten verschiedene Modelle, damit du mobil sein kannst: [zutreffende Optionen ergänzen – wie Bahncard, Dienstfahrrad oder Firmenwagen].
 Sabbatical oder Elternzeit werden unterstützt. Sie sind bei uns kein Karriere-Stopper!
 Wir sind an deiner Seite, auch wenn es einmal nicht so gut läuft: Du kannst Sonderurlaub nehmen, und unsere Beratungshotline steht dir immer zur Verfügung.
 Eine Vielzahl an gemeinsamen Events und Freizeitaktivitäten stärkt deine Verbundenheit mit deinen Kolleg:innen.

Together, as owners, let’s turn meaningful insights into action.

1976 gegründet und nach wie vor familiengeführt, ist CGI heute einer der weltweit größten unabhängigen Anbieter von IT und Business Consulting. Ein hohes Maß an Eigenverantwortung, Teamwork, Respekt und Zusammenhalt machen das Arbeiten bei uns besonders. Bei uns kannst du dein volles Potenzial entfalten!

Du darfst dich vom ersten Tag an als Miteigentümer:in von CGI verstehen. Wir lassen unsere Vision gemeinsam Wirklichkeit werden. Wir profitieren von unserem gemeinsamen Erfolg und haben die Möglichkeit und die Verantwortung, die Strategie und Ausrichtung unseres Unternehmens aktiv mitzugestalten.

Deine Arbeit schafft Mehrwert. Du findest innovative Lösungen und stärkst dein Netzwerk aus Kolleg:innen und Kunden. Gleichzeitig hast du Zugang zu globalen Ressourcen, um große Ideen zu verwirklichen, neue Chancen zu nutzen und von der immensen Branchen- und Technologie-Kompetenz zu profitieren.

Du bringst deine Karriere voran, da du in einem Unternehmen arbeitest, das auf Wachstum und Langlebigkeit ausgelegt ist. Du wirst von Führungskräften unterstützt, die deine Gesundheit und Zufriedenheit fördern – und dir Möglichkeiten bieten, deine Fähigkeiten zu vertiefen und deinen Horizont zu erweitern.

    """
    
    print("🚀 Job Analyzer Test")
    print("=" * 50)
    
    # Option 1: Quantized model (recommended for most systems)
    print("\n1️⃣  Testing with quantized model (faster, less memory):")
    analyzer_q = JobAnalyzer(
                            model="mistral:7b-instruct-q4_0",
                            use_quantized=True
                            )
    
    # Option 2: Full model (uncomment if you have enough VRAM)
    # print("\n2️⃣  Testing with full model (slower, more memory):")
    # analyzer_full = JobAnalyzer(use_quantized=False)
    
    try:
        analysis = analyzer_q.analyze_job_posting(sample_job)
        analyzer_q.print_analysis(analysis)
        
        # Save to file for testing
        with open("sample_analysis.json", "w") as f:
            json.dump(analysis.model_dump(), f, indent=2)
        print(f"\n💾 Analysis saved to sample_analysis.json")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Install the model: ollama pull llama3.2:3b-instruct-q4_K_M")
        print("3. Check if model is available: ollama list")