@echo off
REM =====================================================================
REM Скрипт загрузки датасета OrganCMNIST
REM =====================================================================

echo.
echo ===================================================================
echo Загрузка датасета OrganCMNIST
echo ===================================================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден! Сначала запустите install_windows.bat
    pause
    exit /b 1
)

echo Датасет будет загружен в: %USERPROFILE%\.medmnist\
echo Размер: ~100 MB
echo.

REM Создаем директорию если не существует
if not exist "%USERPROFILE%\.medmnist" (
    mkdir "%USERPROFILE%\.medmnist"
    echo [OK] Создана директория %USERPROFILE%\.medmnist\
)

REM Проверка существования датасета
if exist "%USERPROFILE%\.medmnist\organcmnist.npz" (
    echo.
    echo [ВНИМАНИЕ] Датасет уже существует!
    echo Пропускаем загрузку...
    echo.
    goto :check_dataset
)

echo Загрузка датасета через Python...
echo.

REM Пытаемся загрузить через medmnist
python -c "from medmnist import OrganCMNIST; import os; print('Загрузка...'); train = OrganCMNIST(split='train', download=True, root=os.path.expanduser('~/.medmnist')); print('[OK] Датасет загружен!')"

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Автоматическая загрузка не удалась!
    echo.
    echo FALLBACK: Загрузка вручную через curl...
    echo.

    REM Fallback: загрузка через curl (доступен в Windows 10+)
    curl --version >nul 2>&1
    if errorlevel 1 (
        echo [ОШИБКА] curl не найден!
        echo.
        echo Скачайте датасет вручную:
        echo 1. Перейдите: https://zenodo.org/records/10519652/files/organcmnist.npz?download=1
        echo 2. Сохраните файл как: %USERPROFILE%\.medmnist\organcmnist.npz
        echo.
        pause
        exit /b 1
    )

    echo Загрузка через curl...
    curl -L "https://zenodo.org/records/10519652/files/organcmnist.npz?download=1" -o "%USERPROFILE%\.medmnist\organcmnist.npz" --ssl-no-revoke

    if errorlevel 1 (
        echo.
        echo [ОШИБКА] Загрузка не удалась!
        echo Скачайте датасет вручную по инструкции выше
        pause
        exit /b 1
    )
)

:check_dataset
echo.
echo ===================================================================
echo Проверка датасета...
echo ===================================================================
echo.

python -c "import os; path = os.path.expanduser('~/.medmnist/organcmnist.npz'); print(f'Файл: {path}'); import os.path; size = os.path.getsize(path) / (1024*1024); print(f'Размер: {size:.2f} MB')"

if errorlevel 1 (
    echo [ОШИБКА] Датасет поврежден или отсутствует!
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo [УСПЕХ] Датасет готов к использованию!
echo ===================================================================
echo.
echo Следующий шаг: Запустите run_baseline.bat для обучения baseline модели
echo Или запустите jupyter_notebook.bat для работы с ноутбуком
echo.
pause
