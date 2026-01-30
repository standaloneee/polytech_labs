@echo off
REM =====================================================================
REM Главный скрипт запуска - START
REM =====================================================================

:menu
cls
echo.
echo ===================================================================
echo   МНОГОСЛОЙНЫЙ ПЕРЦЕПТРОН - КЛАССИФИКАЦИЯ ORGANCMNIST
echo ===================================================================
echo.
echo   Выберите действие:
echo.
echo   1. Установить зависимости (первый запуск)
echo   2. Загрузить датасет
echo   3. Запустить Jupyter Notebook
echo   4. Обучить baseline модель
echo   5. Запустить эксперименты (быстрые, 6 шт, ~8 мин)
echo   6. Запустить все эксперименты (24 шт, ~40 мин)
echo   7. Посмотреть результаты
echo   8. Открыть документацию
echo   0. Выход
echo.
echo ===================================================================
echo.

set /p choice="Введите номер (0-8): "

if "%choice%"=="1" goto :install
if "%choice%"=="2" goto :download
if "%choice%"=="3" goto :jupyter
if "%choice%"=="4" goto :baseline
if "%choice%"=="5" goto :quick
if "%choice%"=="6" goto :full
if "%choice%"=="7" goto :results
if "%choice%"=="8" goto :docs
if "%choice%"=="0" goto :exit

echo.
echo [ОШИБКА] Неверный выбор!
timeout /t 2 >nul
goto :menu

:install
call scripts\windows\install_windows.bat
echo.
echo Нажмите любую клавишу для возврата в меню...
pause >nul
goto :menu

:download
call scripts\windows\download_dataset.bat
echo.
echo Нажмите любую клавишу для возврата в меню...
pause >nul
goto :menu

:jupyter
call scripts\windows\jupyter_notebook.bat
echo.
echo Нажмите любую клавишу для возврата в меню...
pause >nul
goto :menu

:baseline
call scripts\windows\run_baseline.bat
echo.
echo Нажмите любую клавишу для возврата в меню...
pause >nul
goto :menu

:quick
echo.
echo Запуск быстрых экспериментов (6 конфигураций × 10 эпох)...
echo Время: ~8 минут
echo.
pause
cd /d "%~dp0"
python -u scripts\python\experiments_quick.py
echo.
echo Результаты: results/experiments_results/quick_test_summary.txt
pause
goto :menu

:full
echo.
echo Запуск полных экспериментов (24 конфигурации × 20 эпох)...
echo Время: ~40 минут
echo.
echo ВНИМАНИЕ: Это займет много времени!
echo.
set /p confirm="Продолжить? (y/n): "
if /i not "%confirm%"=="y" goto :menu

cd /d "%~dp0"
python -u scripts\python\experiments.py
echo.
echo Результаты: results/experiments_results/summary.txt
pause
goto :menu

:results
echo.
echo Открытие папки с результатами...
start "" "%~dp0results"
timeout /t 2 >nul
goto :menu

:docs
echo.
echo Доступная документация:
echo.
echo   1. README.md - общее описание проекта
echo   2. docs/guide.md - подробный технический гайд
echo   3. docs/final_report.txt - итоговый отчет
echo   4. docs/WINDOWS_GUIDE.md - инструкция для начинающих
echo   5. docs/GPU_SETUP.md - настройка GPU (CUDA/MPS)
echo.
echo Открыть в блокноте? (1/2/3/4/5/0-отмена)
set /p doc="Выберите документ: "

if "%doc%"=="1" notepad README.md
if "%doc%"=="2" notepad docs\guide.md
if "%doc%"=="3" notepad docs\final_report.txt
if "%doc%"=="4" notepad docs\WINDOWS_GUIDE.md
if "%doc%"=="5" notepad docs\GPU_SETUP.md

goto :menu

:exit
echo.
echo Выход из программы...
exit /b 0
