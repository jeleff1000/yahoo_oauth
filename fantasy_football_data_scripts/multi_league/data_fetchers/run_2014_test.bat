@echo off
echo Running 2014 data fetch diagnostic...
cd C:\Users\joeye\OneDrive\Desktop\KMFFLApp
C:\Users\joeye\OneDrive\Desktop\KMFFLApp\.venv\Scripts\python.exe test_2014_fetch.py > test_2014_output.txt 2>&1
echo.
echo Diagnostic complete. Results saved to test_2014_output.txt
echo.
type test_2014_output.txt
pause

