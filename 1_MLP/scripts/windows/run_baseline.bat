@echo off
REM =====================================================================
REM Скрипт обучения baseline модели
REM =====================================================================

echo.
echo ===================================================================
echo Обучение Baseline модели MLP
echo ===================================================================
echo.

REM Проверка датасета
if not exist "%USERPROFILE%\.medmnist\organcmnist.npz" (
    echo [ОШИБКА] Датасет не найден!
    echo Сначала запустите download_dataset.bat
    pause
    exit /b 1
)

echo Конфигурация Baseline:
echo  - Архитектура: [128, 64]
echo  - Эпох: 20
echo  - Без регуляризации
echo  - Ожидаемая точность: ~74-75%%
echo  - Время обучения: ~5-10 минут
echo.
echo Нажмите Enter для начала обучения...
pause >nul

cd /d "%~dp0"
cd ..\..
python scripts\python\train_mlp.py

if errorlevel 1 (
    echo.
    echo [ОШИБКА] Обучение завершилось с ошибкой!
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo [УСПЕХ] Обучение завершено!
echo ===================================================================
echo.
echo Результаты сохранены в: results/baseline_results/
echo  - baseline_training.png - графики обучения
echo  - baseline_model.pth - веса модели
echo  - baseline_results.txt - текстовый отчет
echo.
echo Следующий шаг: run_experiments.bat для запуска экспериментов
echo.
pause
