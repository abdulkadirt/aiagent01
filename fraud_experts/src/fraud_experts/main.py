#!/usr/bin/env python
import sys
import warnings
from datetime import datetime
from pathlib import Path

from fraud_experts.crew import FraudExperts

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


# =============================================================================
# FRAUD DETECTION CREW - MAIN ENTRY POINT
# =============================================================================

def run():
    """
    Fraud detection crew'unu çalıştır.

    Veri Stratejisi:
    - İlk aşamada: 50,000 satırlık örneklem (hızlı geliştirme)
    - Final aşamada: Tüm veri (production run)
    """

    # ==========================================================================
    # PATH CONFIGURATION - Windows path sorunlarından kaçınmak için Path kullan
    # ==========================================================================

    # Proje ana klasörü
    project_root = Path(__file__).parent.parent.parent  # main.py'den 3 üst klasör

    # Veri klasörleri
    processed_data_dir = project_root / "data" / "processed"
    train_file = processed_data_dir / "train_merged.csv"

    # Çıktı klasörleri (oluştur)
    features_dir = project_root / "data" / "features"
    models_dir = project_root / "data" / "models"
    reports_dir = project_root / "data" / "reports"

    features_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # DATA SAMPLING CONFIGURATION
    # ==========================================================================
    # Geliştirme aşamasında: sample_size = 50000 (hızlı)
    # Production aşamasında: sample_size = None (tüm veri)
    # ==========================================================================

    USE_SAMPLE = True  # False yap → tüm veriyle çalış
    SAMPLE_SIZE = 50000  # İlk 50k satır

    # ==========================================================================
    # INPUT PARAMETERS
    # ==========================================================================

    inputs = {
        # Veri dosyasının tam yolu (string olarak)
        'dataset_path': str(train_file),

        # Örneklem kullanılacak mı?
        'use_sample': USE_SAMPLE,
        'sample_size': SAMPLE_SIZE if USE_SAMPLE else None,

        # Çıktı klasörleri
        'features_dir': str(features_dir),
        'models_dir': str(models_dir),
        'reports_dir': str(reports_dir),

        # Güncel yıl
        'current_year': str(datetime.now().year)
    }

    # ==========================================================================
    # VALIDATION - Dosya var mı kontrol et
    # ==========================================================================

    if not train_file.exists():
        print("=" * 80)
        print("❌ ERROR: Train data file not found!")
        print(f"   Expected location: {train_file}")
        print()
        print("   Please ensure you have:")
        print("   1. Downloaded IEEE-CIS data from Kaggle")
        print("   2.  Merged train_transaction.csv + train_identity.csv")
        print("   3.  Saved as: data/processed/train_merged.csv")
        print("=" * 80)
        return

    # ==========================================================================
    # START CREW
    # ==========================================================================

    print("=" * 80)
    print("🚀 Starting Fraud Detection Crew")
    print("=" * 80)
    print(f"📂 Train Data: {train_file}")
    print(f"📊 Data Size: {train_file.stat().st_size / (1024 ** 3):.2f} GB")

    if USE_SAMPLE:
        print(f"🔬 Mode: DEVELOPMENT (using {SAMPLE_SIZE:,} samples)")
        print("   💡 Set USE_SAMPLE=False in main.py for full dataset run")
    else:
        print(f"🏭 Mode: PRODUCTION (using full dataset)")

    print(f"📅 Current Year: {inputs['current_year']}")
    print("=" * 80)
    print()

    try:
        # Crew'u başlat ve çalıştır
        result = FraudExperts().crew().kickoff(inputs=inputs)

        print()
        print("=" * 80)
        print("✅ Crew execution completed successfully!")
        print("=" * 80)
        print("📄 Output file: fraud_detection_evaluation_report.md")
        print(f"📁 Reports directory: {reports_dir}")
        print("=" * 80)

        return result

    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ An error occurred while running the crew:")
        print(f"   {str(e)}")
        print("=" * 80)
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Crew'u train et (öğrenme için).
    """
    # Path setup aynı şekilde
    project_root = Path(__file__).parent.parent.parent
    train_file = project_root / "data" / "processed" / "train_merged.csv"

    inputs = {
        'dataset_path': str(train_file),
        'use_sample': True,
        'sample_size': 50000,
        'current_year': str(datetime.now().year)
    }

    try:
        FraudExperts().crew().train(
            n_iterations=int(sys.argv[1]),
            filename=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Bir task'ı replay et.
    """
    try:
        FraudExperts().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Crew'u test et.
    """
    project_root = Path(__file__).parent.parent.parent
    train_file = project_root / "data" / "processed" / "train_merged.csv"

    inputs = {
        'dataset_path': str(train_file),
        'use_sample': True,
        'sample_size': 50000,
        'current_year': str(datetime.now().year)
    }

    try:
        FraudExperts().crew().test(
            n_iterations=int(sys.argv[1]),
            eval_llm=sys.argv[2],
            inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")


if __name__ == "__main__":
    run()