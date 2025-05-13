# Hayvan Sınıflandırıcı - Yapay Zeka Destekli Görüntü Tanıma

Bu proje, kullanıcıların yüklediği hayvan görsellerini yapay zeka kullanarak otomatik olarak sınıflandırabilen bir uygulamadır. 10 hayvan sınıfı içeren veri seti kullanılarak eğitilmiş derin öğrenme modeli, farklı hayvan türlerini yüksek doğrulukla tanıyabilmektedir.

## Hızlı Başlangıç Kılavuzu

Projeyi çalıştırmak için izlenecek adımlar:

### 1. Gerekli Kütüphaneleri Yükleme

İlk olarak, gereksinimleri yükleyin:

```bash
# Temel gereksinimleri yükleme
pip install -r requirements.txt

# GPU desteği için (NVIDIA GPU kullanıyorsanız)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Veri Seti Hazırlama

Ham hayvan görüntülerini eğitim ve test setlerine ayırmak için:

```bash
python prepare_animal_dataset.py
```

Bu komut, `data` klasöründeki hayvan görüntülerini alıp `dataset/train` ve `dataset/test` klasörlerine dağıtır.

### 3. Model Eğitimi

Sınıflandırma modelini eğitmek için:

```bash
python train_model.py --num_classes 10 --num_epochs 10
```

- `--num_epochs` parametresi eğitim süresi için ayarlanabilir (daha yüksek değerler daha uzun eğitim süreleri anlamına gelir)
- Eğitim sonunda, model ağırlıkları `model/model_weights.pth` dosyasına, sınıf eşleştirmeleri `model/class_mapping.json` dosyasına ve metrikler `model/metrics.json` dosyasına kaydedilir.

### 4. Web Uygulamasını Başlatma

Modeli kullanmak için Streamlit web uygulamasını başlatın:

```bash
python -m streamlit run app.py
```

Tarayıcınızda otomatik olarak bir sayfa açılacaktır (genellikle http://localhost:8501). Bu sayfada:
1. "Sınıflandırmak için bir hayvan görüntüsü yükleyin" butonuna tıklayın
2. Bir görüntü seçin
3. "Tahmin Et" butonuna tıklayarak sınıflandırma sonucunu alın

### 5. Alternatif: Hazır Script ile Çalıştırma

Tüm adımları (kurulum, veri hazırlama, eğitim ve çalıştırma) tek bir script ile gerçekleştirmek için:

**Windows için:**
```bash
# Sadece web arayüzünü başlatmak için
run_example.bat

# Veri seti hazırlama ve model eğitimini de gerçekleştirmek için
run_example.bat -train

# Tüm parametrelerle kullanmak için (-train bayrağı ile)
run_example.bat -train 10 15 32 resnet50
```

**Linux/Mac için:**
```bash
./run_example.sh
```

## Özellikler

- Hayvan görüntüsü yükleme ve önizleme
- Otomatik görüntü önişleme (boyutlandırma, normalizasyon)
- Derin öğrenme modeli ile hayvan türü tanıma
- Kullanıcı dostu web arayüzü
- Sınıflandırma sonuçlarının görsel olarak sunumu
- En olası hayvan türlerinin yüzdelik olasılıklarla gösterimi
- GPU hızlandırma desteği (CUDA)

## Veri Seti Bilgileri

Bu projede 10 hayvan sınıfı içeren bir veri seti kullanılmıştır. Tanınan hayvan türleri:

1. **cane** - köpek
2. **cavallo** - at
3. **elefante** - fil
4. **farfalla** - kelebek
5. **gallina** - tavuk
6. **gatto** - kedi
7. **mucca** - inek
8. **pecora** - koyun
9. **ragno** - örümcek
10. **scoiattolo** - sincap

## Kurulum

1. Projeyi bilgisayarınıza klonlayın:
```
git clone https://github.com/ibrahimhates/ai_animal_image_recognition
cd hayvan-siniflandirici
```

2. Gerekli kütüphaneleri yükleyin:
```
pip install -r requirements.txt
```

3. (Opsiyonel) GPU Desteği:
   - NVIDIA GPU'nuz varsa, CUDA ile çalışacak PyTorch sürümünü yükleyebilirsiniz:
```bash
# Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Linux/Mac
# Linux genellikle CUDA sürümüne özel komut gerektirir
# Mac'te ise MPS (Metal Performance Shaders) kullanılır
pip install torch torchvision torchaudio
```

4. Veri setini hazırlamak için:
```
python prepare_animal_dataset.py
```
Bu komut, 'data' klasöründeki hayvan görüntülerini alır ve bunları eğitim ve test için ayırarak 'dataset' klasöründe düzenler.

## Kullanım

### Model Eğitimi

Modeli eğitmek için (eğer önceden eğitilmiş model yoksa):

```
python train_model.py --num_classes [HAYVAN_TURU_SAYISI] --num_epochs [EPOCH_SAYISI] --batch_size [BATCH_SIZE] --model_name [MODEL_ADI]
```

#### Parametre Açıklamaları:

| Parametre | Açıklama | Varsayılan Değer |
|------------|---------|-----------------|
| `--num_classes` | Sınıflandırılacak hayvan türü sayısı | 10 |
| `--num_epochs` | Eğitim döngüsü (epoch) sayısı | 10 |
| `--batch_size` | Toplu işleme boyutu | 32 |
| `--model_name` | Kullanılacak model mimarisi | resnet18 |
| `--learning_rate` | Öğrenme oranı | 0.001 |
| `--data_dir` | Veri seti klasörü | dataset |
| `--save_path` | Model kayıt yolu | model/model_weights.pth |

#### Örnek Kullanımlar:

```bash
# Temel kullanım (10 hayvan türü, 10 epoch)
python train_model.py

# Daha kısa eğitim (10 hayvan türü, 5 epoch)
python train_model.py --num_epochs 5

# Farklı model mimarisi kullanımı
python train_model.py --model_name mobilenet_v2
```

### Hazır Komut Dosyaları ile Kullanım

Projede hazır komut dosyaları bulunmaktadır:

#### Windows için:

```bash
# Temel kullanım (Sadece web arayüzünü başlatır)
run_example.bat

# Eğitim modu (Veri seti hazırlama ve eğitim gerçekleştirir)
run_example.bat -train [SINIF_SAYISI] [EPOCH_SAYISI] [BATCH_SIZE] [MODEL_ADI]
```

Örnekler:
```bash
# Sadece web arayüzünü başlatma
run_example.bat

# Veri seti hazırlama ve varsayılan ayarlarla eğitim
run_example.bat -train

# Veri seti hazırlama ve özel parametrelerle eğitim
run_example.bat -train 10 5 64 resnet34
```

#### Linux/Mac için:

```bash
./run_example.sh [SINIF_SAYISI] [EPOCH_SAYISI] [BATCH_SIZE] [MODEL_ADI]
```

Örnekler:
```bash
# Varsayılan değerlerle (10 sınıf, 10 epoch) - GPU otomatik tespit edilir
./run_example.sh

# 10 sınıf, 5 epoch ile
./run_example.sh 10 5

# Tam parametre kullanımı
./run_example.sh 10 10 32 resnet18
```

### Web Arayüzü

Uygulamayı başlatmak için:

```
python -m streamlit run app.py
```

Tarayıcıda otomatik olarak uygulama arayüzü açılacaktır. "Görüntü Yükle" butonuna tıklayarak bir hayvan resmi seçebilir ve "Tahmin Et" butonuna basarak seçilen hayvanın türünü belirleyebilirsiniz.

## GPU Desteği

Bu uygulama, eğitim ve tahmin işlemlerini hızlandırmak için GPU desteği içerir:

- **NVIDIA GPU (CUDA)**: Windows ve Linux sistemlerde NVIDIA GPU'lar CUDA aracılığıyla kullanılır
- **Apple Silicon (MPS)**: M1/M2/M3 işlemcili Mac bilgisayarlarda Metal Performance Shaders kullanılır
- **Otomatik Tespit**: Model GPU kullanılabilirliğini otomatik olarak tespit eder ve uygun cihazda çalışır

GPU kullanımını kontrol etmek için:
```python
python -c "import torch; print('CUDA Kullanılabilir:', torch.cuda.is_available())"
```

## Model Çıktıları ve Sonuçları

Eğitim sonrası, aşağıdaki dosyalar oluşturulur:

### 1. Model Ağırlıkları (`model/model_weights.pth`)
- Eğitilmiş modelin ağırlıklarını ve parametrelerini içerir
- PyTorch model dosyasıdır
- Modelin tahmin yapabilmesi için gereklidir

### 2. Sınıf Eşleştirme (`model/class_mapping.json`)
- Modelin tahmin ettiği sayısal indekslerin hangi hayvan türüne karşılık geldiğini belirten JSON dosyası
- Örnek: `{"0": "cane", "1": "cavallo", ...}`

### 3. Metrikler (`model/metrics.json`)
- Modelin performans metriklerini içeren JSON dosyası
- İçerdiği metrikler:
  - **Accuracy (Doğruluk)**: Modelin doğru tahmin oranı
  - **Precision (Kesinlik)**: Pozitif tahminlerin ne kadarının gerçekten pozitif olduğu
  - **Recall (Duyarlılık)**: Gerçek pozitiflerin ne kadarının doğru tahmin edildiği

Bu dosyalar, web uygulaması çalıştırıldığında otomatik olarak kullanılır. Eğer bu dosyalar yoksa veya bozulmuşsa, model eğitimi yeniden yapılmalıdır.

## Sorun Giderme

### Model Yüklenme Hataları
- **"Error(s) in loading state_dict"**: Model boyutu uyumsuzluğu, genellikle farklı bir sınıf sayısı ile eğitilmiş olduğunu gösterir. Çözüm için `train_model.py` ile doğru sınıf sayısını kullanarak modeli yeniden eğitin.
- **"Model yüklenirken hata oluştu"**: Model dosyaları eksik veya bozuk olabilir. Eğitimi yeniden çalıştırın.

### Veri Seti Sorunları
- **"Kaynak dizini bulunamadı"**: `data` klasöründeki hayvan sınıflarını kontrol edin.
- **"Algılanan hayvan türü sayısı: 0"**: Veri seti klasörleri boş olabilir. Görüntülerin doğru klasörlerde olduğundan emin olun.

### GPU Sorunları
- **CUDA hatası**: GPU sürücülerinizi güncelleyin veya CPU moduna geçin.

## Proje Yapısı

```
hayvan-siniflandirici/
├── app.py                  # Streamlit web arayüzü
├── requirements.txt        # Gerekli kütüphaneler
├── train_model.py          # Model eğitim başlatma scripti
├── prepare_animal_dataset.py # Veri seti hazırlama scripti
├── run_example.bat         # Windows için çalıştırma dosyası
├── run_example.sh          # Linux/Mac için çalıştırma dosyası
├── model/                  # Model dosyaları
│   ├── train.py            # Model eğitim kodu
│   ├── model.py            # Model mimarisi
│   ├── utils.py            # Yardımcı fonksiyonlar
│   └── model_weights.pth   # Eğitilmiş model ağırlıkları
├── data/                   # Ham hayvan görüntüleri
│   ├── cane/               # Köpek görüntüleri
│   ├── cavallo/            # At görüntüleri
│   └── ...                 # Diğer hayvan görüntüleri
├── dataset/                # İşlenmiş veri seti
│   ├── train/              # Eğitim veri seti
│   └── test/               # Test veri seti
└── preprocessing/          # Görüntü ön işleme modülü
    └── preprocessing.py    # Ön işleme fonksiyonları
``` 
