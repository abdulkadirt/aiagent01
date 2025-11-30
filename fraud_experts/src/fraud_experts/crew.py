from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
import os
from dotenv import load_dotenv

# . env dosyasını yükle (MODEL ve GEMINI_API_KEY gibi değişkenler için)
load_dotenv()


# =============================================================================
# FRAUD DETECTION CREW
# =============================================================================
# Bu crew, fraud detection için 5 agent ve 5 task içerir.
# Agent'lar sırayla çalışır (sequential process):
# Research → Data Analysis → Feature Engineering → Model Development → Evaluation
# =============================================================================

@CrewBase
class FraudExperts():
    """FraudExperts crew for IEEE-CIS fraud detection analysis"""

    # CrewAI otomatik olarak bu listeleri dolduracak
    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        """
        Crew başlatılırken LLM'i yapılandır.

        Ne yapıyor?
        - . env dosyasından model adını ve API key'ini okur
        - Tüm agent'ların kullanacağı LLM nesnesini oluşturur
        - Temperature ile yaratıcılık seviyesini ayarlar

        Neden __init__ içinde?
        - Crew nesnesi her oluşturulduğunda bir kez çalışır
        - Merkezi bir yerden LLM ayarlarını yönetiriz
        """
        # . env'den model adını oku, yoksa default değer kullan
        model_name = os.getenv("MODEL", "gemini/gemini-2.5-flash")
        api_key = os.getenv("GEMINI_API_KEY")

        # LLM nesnesini oluştur
        self.llm = LLM(
            model=model_name,
            api_key=api_key,
            temperature=0.7  # 0=deterministik, 1=yaratıcı (fraud için 0.7 dengeli)
        )

        print(f"🤖 LLM Configured: {model_name}")  # Debug için

    # =========================================================================
    # AGENTS - Her agent metodu @agent decorator'ı ile işaretlenir
    # =========================================================================
    # Neden metot isimleri agents.yaml'daki key'lerle aynı?
    # - CrewAI otomatik olarak config=self.agents_config['fraud_research_agent']
    #   şeklinde YAML'dan config çeker
    # - İsimlendirme tutarlılığı zorunlu!
    # =========================================================================

    @agent
    def fraud_research_agent(self) -> Agent:
        """
        Fraud detection yöntemlerini araştıran agent.

        Neden verbose=True?
        - Agent'ın düşünce sürecini görmek için (debug/öğrenme amaçlı)
        - Production'da False yapılabilir

        Neden llm=self.llm?
        - __init__'de tanımladığımız LLM'i kullanır
        - Tüm agent'lar aynı model kullanır (tutarlılık)
        """
        return Agent(
            config=self.agents_config['fraud_research_agent'],
            verbose=True,
            llm=self.llm
        )

    @agent
    def data_analyst_agent(self) -> Agent:
        """
        Veri setini analiz eden agent (EDA yapan).
        """
        return Agent(
            config=self.agents_config['data_analyst_agent'],
            verbose=True,
            llm=self.llm
        )

    @agent
    def feature_engineer_agent(self) -> Agent:
        """
        Feature'ları tasarlayan agent.
        """
        return Agent(
            config=self.agents_config['feature_engineer_agent'],
            verbose=True,
            llm=self.llm
        )

    @agent
    def ml_engineer_agent(self) -> Agent:
        """
        Model geliştiren ve eğiten agent.
        """
        return Agent(
            config=self.agents_config['ml_engineer_agent'],
            verbose=True,
            llm=self.llm
        )

    @agent
    def model_evaluator_agent(self) -> Agent:
        """
        Modeli değerlendiren ve iyileştirme öneren agent.
        """
        return Agent(
            config=self.agents_config['model_evaluator_agent'],
            verbose=True,
            llm=self.llm
        )

    # =========================================================================
    # TASKS - Her task metodu @task decorator'ı ile işaretlenir
    # =========================================================================
    # Task'lar sequential olarak çalışır (yukarıdan aşağıya doğru)
    # Her task, tasks.yaml'dan config çeker
    # =========================================================================

    @task
    def research_fraud_methods_task(self) -> Task:
        """
        TASK 1: Fraud detection yöntemlerini araştır.

        Bu task:
        - Kaggle'da IEEE-CIS yarışmasını araştırır
        - En iyi teknikleri bulur
        - Actionable öneriler listesi oluşturur

        Bağımlılık: Yok (ilk task)
        Agent: fraud_research_agent
        """
        return Task(
            config=self.tasks_config['research_fraud_methods_task'],
        )

    @task
    def data_analysis_task(self) -> Task:
        """
        TASK 2: Veri setini analiz et (EDA).

        Bu task:
        - CSV dosyalarını okur
        - Missing values, distributions, correlations analiz eder
        - Fraud pattern'leri bulur
        - EDA raporu oluşturur

        Bağımlılık: research_fraud_methods_task (araştırma bulgularını kullanır)
        Agent: data_analyst_agent
        """
        return Task(
            config=self.tasks_config['data_analysis_task'],
        )

    @task
    def feature_engineering_task(self) -> Task:
        """
        TASK 3: Feature mühendisliği planı oluştur.

        Bu task:
        - Temporal, aggregation, interaction feature'ları tasarlar
        - Data leakage kontrolü yapar
        - Kod template'leri sağlar

        Bağımlılık: data_analysis_task, research_fraud_methods_task
        Agent: feature_engineer_agent
        """
        return Task(
            config=self.tasks_config['feature_engineering_task'],
        )

    @task
    def model_development_task(self) -> Task:
        """
        TASK 4: Model geliştir ve eğit.

        Bu task:
        - Model seçer (XGBoost, LightGBM)
        - Class imbalance handling yapar
        - Hyperparameter tuning önerir
        - Training pipeline kodu sağlar

        Bağımlılık: feature_engineering_task, data_analysis_task, research_fraud_methods_task
        Agent: ml_engineer_agent
        """
        return Task(
            config=self.tasks_config['model_development_task'],
        )

    @task
    def model_evaluation_task(self) -> Task:
        """
        TASK 5: Modeli değerlendir ve iyileştir.

        Bu task:
        - AUC-ROC, PR-AUC, confusion matrix hesaplar
        - Business impact analizi yapar
        - Error analysis yapar
        - İyileştirme önerileri verir
        - Sonucu fraud_detection_evaluation_report.md'ye yazar

        Bağımlılık: Tüm önceki task'lar
        Agent: model_evaluator_agent
        Output: fraud_detection_evaluation_report.md dosyası
        """
        return Task(
            config=self.tasks_config['model_evaluation_task'],
            output_file='fraud_detection_evaluation_report.md'  # Sonuç dosyası
        )

    # =========================================================================
    # CREW - Tüm agent'ları ve task'ları bir araya getirir
    # =========================================================================

    @crew
    def crew(self) -> Crew:
        """
        FraudExperts Crew'unu oluşturur.

        Ne yapar?
        - Tüm agent'ları toplar (self.agents)
        - Tüm task'ları toplar (self.tasks)
        - Sequential process ile sırayla çalışır

        Process tipleri:
        - sequential: Task'lar sırayla çalışır (bizim durumumuz)
        - hierarchical: Manager agent diğerlerini yönetir (daha karmaşık)

        Neden sequential?
        - Her task bir öncekine bağımlı (Research → Analysis → Features → Model → Eval)
        - Paralel çalışma mantıklı değil
        """
        return Crew(
            agents=self.agents,  # @agent ile işaretlenmiş tüm metotlar
            tasks=self.tasks,  # @task ile işaretlenmiş tüm metotlar
            process=Process.sequential,  # Sıralı çalışma
            verbose=True,  # Detaylı log
        )