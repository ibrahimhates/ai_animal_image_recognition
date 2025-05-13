#!/bin/bash

# Bu dosya, projeyi çalıştırmak için örnek komutları içerir.
# Windows'ta PowerShell veya komut isteminde çalıştırmak isterseniz aşağıdaki komutları ayrı ayrı çalıştırın.

# Varsayılan parametreler
NUM_CLASSES=10
NUM_EPOCHS=10
BATCH_SIZE=32
MODEL_NAME="resnet18"

# Komut satırı parametrelerini kontrol et
if [ ! -z "$1" ]; then
    NUM_CLASSES=$1
fi

if [ ! -z "$2" ]; then
    NUM_EPOCHS=$2
fi

if [ ! -z "$3" ]; then
    BATCH_SIZE=$3
fi

if [ ! -z "$4" ]; then
    MODEL_NAME=$4
fi

echo "Kullanılacak Parametreler:"
echo "Sınıf Sayısı: $NUM_CLASSES"
echo "Epoch Sayısı: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Model: $MODEL_NAME"

# 1. Gerekli kütüphaneleri yükle
echo "Gerekli kütüphaneleri yükleme..."
pip install -r requirements.txt

# GPU destekli PyTorch yükleme 
echo "GPU destekli PyTorch kurulumu yapılıyor..."
# Linux/Mac için farklı kurulum komutu kullanılabilir
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # Linux için CUDA 11.8
pip install torch torchvision torchaudio # Mac/Linux için otomatik en uygun versiyonu seçer

# Streamlit'in doğru yüklendiğinden emin ol
echo "Streamlit kurulumunu kontrol ediyorum..."
pip install streamlit

# GPU durumunu kontrol et
echo "GPU durumu kontrol ediliyor..."
python3 -c "import torch; print('CUDA Kullanılabilir:', torch.cuda.is_available()); print('GPU Sayısı:', torch.cuda.device_count()); print('GPU Adı:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Yok')"

# Dataset oluşturma
echo "Veri seti hazırlanıyor..."
python prepare_animal_dataset.py

# 2. Modeli eğit (eğer önceden eğitilmiş bir model yoksa)
# NOT: Veri setiniz kurulmuş olmalıdır ve 'dataset' klasöründe bulunmalıdır.
echo "Hayvan sınıflandırma modeli eğitimi..."

# Veri setinin sınıf sayısını kontrol et
CLASS_COUNT=$(find dataset/train -type d | wc -l)
CLASS_COUNT=$((CLASS_COUNT - 1))  # Ana klasörü çıkar
echo "Algılanan hayvan türü sayısı: $CLASS_COUNT"

# Eğitim komutu
echo "Eğitim başlatılıyor..."
python train_model.py --num_classes $NUM_CLASSES --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --model_name $MODEL_NAME

# 3. Web arayüzünü başlat
echo "Web arayüzünü başlatma..."
python -m streamlit run app.py 