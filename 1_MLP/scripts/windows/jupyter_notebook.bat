@echo off
REM =====================================================================
REM Скрипт запуска Jupyter Notebook
REM =====================================================================

echo.
echo ===================================================================
echo Запуск Jupyter Notebook
echo ===================================================================
echo.

REM Проверка Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Python не найден! Сначала запустите install_windows.bat
    pause
    exit /b 1
)

REM Проверка Jupyter
python -m jupyter --version >nul 2>&1
if errorlevel 1 (
    echo [ОШИБКА] Jupyter не установлен!
    echo Установка Jupyter...
    python -m pip install jupyter
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить Jupyter
        pause
        exit /b 1
    )
)

REM Проверка датасета
if not exist "%USERPROFILE%\.medmnist\organcmnist.npz" (
    echo.
    echo [ВНИМАНИЕ] Датасет не найден!
    echo Сначала запустите download_dataset.bat
    echo.
    pause
    exit /b 1
)

echo [OK] Все проверки пройдены
echo.
echo ===================================================================
echo Запуск Jupyter Notebook...
echo ===================================================================
echo.
echo ВАЖНО:
echo  - Jupyter откроется в браузере автоматически
echo  - Откройте файл: notebooks/mlp_organcmnist.ipynb
echo  - Для остановки: нажмите Ctrl+C в этом окне
echo.
echo Нажмите Enter для продолжения...
pause >nul

REM Запуск Jupyter в директории проекта
cd /d "%~dp0"
cd ..\..
python -m jupyter notebook notebooks/mlp_organcmnist.ipynb

pause
