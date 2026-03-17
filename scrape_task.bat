@echo off
cd /d C:\Users\Rn5ho\Tennis-pm

:: Make sure we're on dev
git checkout dev >nul 2>&1

:: Run the scraper
C:\Python313\python.exe run_scraper.py >> data\scraper.log 2>&1

:: Auto-commit and push any changes to GitHub
git add -A >nul 2>&1
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "auto: scraper data update %date% %time%" >nul 2>&1
    git push >nul 2>&1
)
