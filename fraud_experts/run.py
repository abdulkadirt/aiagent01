"""
Manuel çalıştırıcı - Fraud Experts
Konum: fraud_experts/run.py (ana klasör)
Çalıştır: python run.py
"""
import sys
import os
from dotenv import load_dotenv

# .env dosyasını yükle
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# API key kontrol
api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
if api_key:
    print(f"✅ API Key bulundu: {api_key[:10]}...")
    os.environ['GOOGLE_API_KEY'] = api_key
    os.environ['GEMINI_API_KEY'] = api_key
else:
    print("❌ API Key bulunamadı!")
    print("Kontrol: .env dosyasında GOOGLE_API_KEY var mı?")
    sys.exit(1)

# src/ klasörünü Python path'ine ekle
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

print(f"📂 Proje: {project_root}")
print(f"📁 Src: {src_path}\n")

# Şimdi import et ve çalıştır
try:
    from fraud_experts.main import run
    print("✅ Modül yüklendi!\n")
    run()
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("\nKontrol et:")
    print("1. src/fraud_experts/ klasörü var mı?")
    print("2. src/fraud_experts/__init__.py var mı?")
    print("3. src/fraud_experts/main.py var mı?")
    sys.exit(1)
except Exception as e:
    print(f"❌ Çalışma hatası: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)