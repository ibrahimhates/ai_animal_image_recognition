@echo off
REM Bu dosya, Windows'ta projeyi çalıştırmak için örnek komutları içerir.

REM Varsayılan parametreler
SET NUM_CLASSES=10
SET NUM_EPOCHS=10
SET BATCH_SIZE=32
SET MODEL_NAME=resnet18
SET TRAIN_FLAG=0

REM Parametre kontrolü
:PARAM_LOOP
IF "%1"=="" GOTO PARAM_DONE
IF "%1"=="-train" SET TRAIN_FLAG=1 && SHIFT && GOTO PARAM_LOOP
IF NOT "%1"=="" SET NUM_CLASSES=%1 && SHIFT
IF NOT "%1"=="" SET NUM_EPOCHS=%1 && SHIFT
IF NOT "%1"=="" SET BATCH_SIZE=%1 && SHIFT
IF NOT "%1"=="" SET MODEL_NAME=%1 && SHIFT
GOTO PARAM_LOOP
:PARAM_DONE

echo Kullanılacak Parametreler:
echo Sınıf Sayısı: %NUM_CLASSES%
echo Epoch Sayısı: %NUM_EPOCHS%
echo Batch Size: %BATCH_SIZE%
echo Model: %MODEL_NAME%
echo Eğitim Modu: %TRAIN_FLAG%

REM 1. Gerekli kütüphaneleri yükle
echo Gerekli kutuphaneleri yukleme...
pip install -r requirements.txt

REM GPU destekli PyTorch yükleme
echo GPU destekli PyTorch kurulumu yapiliyor...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Streamlit'in doğru yüklendiğinden emin ol
echo Streamlit kurulumunu kontrol ediyorum...
pip install streamlit

REM GPU durumunu kontrol et
echo GPU durumu kontrol ediliyor...
python -c "import torch; print('CUDA Kullanılabilir:', torch.cuda.is_available()); print('GPU Sayısı:', torch.cuda.device_count()); print('GPU Adı:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'Yok')"

REM Dataset oluşturma ve eğitim sadece TRAIN_FLAG=1 ise çalışır
IF %TRAIN_FLAG%==1 (
    REM Dataset oluşturma
    echo Veri seti hazirlaniyor...
    python prepare_animal_dataset.py

    REM 2. Modeli eğit (eğer önceden eğitilmiş bir model yoksa)
    REM NOT: Veri setiniz kurulmuş olmalıdır ve 'dataset' klasöründe bulunmalıdır.

    echo Hayvan siniflandirma modeli egitimi...

    REM Windows'ta klasör sayımı için PowerShell kullanma
    echo Hayvan turu sayisi hesaplaniyor...
    powershell -Command "$classCount = (Get-ChildItem -Path 'dataset\train' -Directory).Count; Write-Host \"Algilanan hayvan turu sayisi: $classCount\""

    REM Kullanıcıya sınıf sayısını sorma (Windows'ta otomatik sayım karmaşık olduğu için)
    SET /P CLASS_COUNT="Yukaridaki hayvan turu sayisini girin: "
    echo %CLASS_COUNT% hayvan turu ile egitim baslatiliyor...

    REM Eğitim komutu
    echo Egitim baslatiliyor...
    python train_model.py --num_classes %NUM_CLASSES% --num_epochs %NUM_EPOCHS% --batch_size %BATCH_SIZE% --model_name %MODEL_NAME%
)

REM 3. Web arayüzünü başlat
echo Web arayuzunu baslatma...
python -m streamlit run app.py 