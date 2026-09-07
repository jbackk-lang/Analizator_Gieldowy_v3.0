@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo ============================================================
echo  Analizator Gieldowy v3 - lokalne API + dashboard
echo  UWAGA: to NIE jest doradztwo inwestycyjne. Narzedzie
echo  badawczo-edukacyjne. Patrz README.md.
echo ============================================================
echo.

rem --- Wykrywanie prawdziwego interpretera (nie aliasu) ---
set "PYCMD="

for %%P in (
    "C:\Users\jback\AppData\Local\Programs\Python\Python312\python.exe"
    "C:\Users\jback\AppData\Local\Programs\Python\Python311\python.exe"
    "python"
    "py"
) do (
    %%~P --version >nul 2>&1
    if not errorlevel 1 (
        set "PYCMD=%%~P"
        goto :found_python
    )
)

echo BLAD: nie znaleziono interpretera Python.
pause
exit /b 1

:found_python
echo Uzywam interpretera: %PYCMD%
%PYCMD% --version
echo.

echo Sprawdzam pip...
%PYCMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo BLAD: pip nie dziala dla tego interpretera.
    echo Sprobuj: %PYCMD% -m ensurepip --upgrade
    pause
    exit /b 1
)

echo.
echo Instalacja/aktualizacja zaleznosci...
%PYCMD% -m pip install --quiet flask numpy pandas yfinance pytest
if errorlevel 1 (
    echo BLAD: instalacja zaleznosci nie powiodla sie.
    pause
    exit /b 1
)

echo.
echo Uruchamiam testy (pytest)...
rem --- basetemp we wlasnym folderze projektu omija zablokowany/uszkodzony
rem     C:\Users\<user>\AppData\Local\Temp\pytest-of-<user> na Windows ---
%PYCMD% -m pytest -q --basetemp=".pytest_tmp"
if errorlevel 1 (
    echo.
    echo UWAGA: co najmniej jeden test nie przeszedl.
    echo.
)

echo.
echo Start serwera na http://127.0.0.1:8060
start "" http://127.0.0.1:8060
%PYCMD% api.py

pause
