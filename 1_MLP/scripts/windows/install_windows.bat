@echo off
REM =====================================================================
REM Скрипт установки зависимостей для Windows
REM =====================================================================

echo.
echo ===================================================================
echo Установка зависимостей для проекта MLP OrganCMNIST
echo ===================================================================
echo.

REM Проверка наличия Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден!
    echo.
    echo Установите Python 3.8+ с официального сайта:
    echo https://www.python.org/downloads/
    echo.
    echo ВАЖНО: При установке отметьте "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo [OK] Python найден:
python --version
echo.

REM Проверка pip
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] pip не найден!
    echo Переустановите Python с включенным pip
    pause
    exit /b 1
)

echo [OK] pip найден
echo.

REM Обновление pip
echo Обновление pip...
python -m pip install --upgrade pip
echo.

REM Установка зависимостей
echo ===================================================================
echo Установка зависимостей из requirements.txt...
echo Это может занять несколько минут
echo ===================================================================
echo.

cd /d "%~dp0"
cd ..\..
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось установить зависимости!
    echo.
    echo Попробуйте установить вручную:
    echo   python -m pip install torch torchvision
    echo   python -m pip install medmnist numpy matplotlib pandas scikit-learn tqdm jupyter
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo [УСПЕХ] Все зависимости установлены!
echo ===================================================================
echo.

REM Проверка установки
echo Проверка установленных пакетов...
echo.
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import medmnist; print(f'MedMNIST: {medmnist.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo.
echo ===================================================================
echo Проверка поддержки GPU...
echo ===================================================================
echo.
python -c "import torch; print('CUDA доступен: ', torch.cuda.is_available()); print('CUDA устройство: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Нет')"

echo.
echo ===================================================================
echo ИНФОРМАЦИЯ О GPU:
echo ===================================================================
echo.
echo Установлена CPU версия PyTorch (по умолчанию).
echo.
echo Если у вас есть NVIDIA GPU и вы хотите ускорить обучение:
echo   1. Установите NVIDIA CUDA Toolkit
echo   2. Переустановите PyTorch с CUDA:
echo      pip uninstall torch torchvision
echo      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo.
echo Скрипты автоматически определят GPU и будут использовать его!
echo CPU версия тоже работает, просто медленнее (~2-3x).
echo.
echo ===================================================================
echo Установка завершена успешно!
echo ===================================================================
echo.
echo Следующий шаг: Запустите download_dataset.bat для загрузки датасета
echo.
pause
