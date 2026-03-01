@echo off
echo ============================================
echo  ST5230 Assignment - Running All Notebooks
echo  Started: %date% %time%
echo ============================================

cd /d "%~dp0"

echo.
echo [1/3] Part 1: Language Model Comparison
echo ----------------------------------------
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/part1_lm_comparison.ipynb
if %errorlevel% neq 0 (
    echo ERROR: Part 1 failed!
    pause
    exit /b 1
)
echo Part 1 done: %time%

echo.
echo [2/3] Part 2: Embedding Ablation
echo ----------------------------------------
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/part2_embedding_ablation.ipynb
if %errorlevel% neq 0 (
    echo ERROR: Part 2 failed!
    pause
    exit /b 1
)
echo Part 2 done: %time%

echo.
echo [3/3] Part 3: Downstream Classification
echo ----------------------------------------
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/part3_downstream.ipynb
if %errorlevel% neq 0 (
    echo ERROR: Part 3 failed!
    pause
    exit /b 1
)
echo Part 3 done: %time%

echo.
echo ============================================
echo  All notebooks finished: %time%
echo ============================================
pause
