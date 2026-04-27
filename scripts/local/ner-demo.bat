@echo off
setlocal EnableDelayedExpansion
rem NER Demo local launcher for Windows — requires Java 21+
rem Downloads the ONNX model (~700 MB) automatically on first run.

set MODEL_URL=https://github.com/pimpmyrag/pimpmyrag/releases/download/v1.0.0-ner-model/best_model_multitask_full.onnx
set CACHE_DIR=%USERPROFILE%\.pimpmyrag
set MODEL_PATH=%CACHE_DIR%\model\best_model_multitask_full.onnx
if not defined NER_PORT set NER_PORT=8090
set JAR=

set SCRIPT_DIR=%~dp0

rem ── Locate JAR ────────────────────────────────────────────────────────────
for %%f in ("%SCRIPT_DIR%*.jar") do (
  echo %%~f | findstr /i /v "plain.jar" >nul 2>&1 && set "JAR=%%~f" && goto :jar_found
)
for %%f in ("%SCRIPT_DIR%..\ner-demo\build\libs\*.jar") do (
  echo %%~f | findstr /i /v "plain.jar" >nul 2>&1 && set "JAR=%%~f" && goto :jar_found
)
:jar_found
if not defined JAR (
  echo ERROR: JAR not found. Download from https://github.com/pimpmyrag/pimpmyrag/releases
  exit /b 1
)

rem ── Check Java ────────────────────────────────────────────────────────────
java -version >nul 2>&1
if errorlevel 1 (
  echo ERROR: Java 21+ required — https://adoptium.net
  exit /b 1
)

rem ── Download model if needed ──────────────────────────────────────────────
if not defined NER_MODEL_PATH (
  if not exist "%MODEL_PATH%" (
    echo Downloading ONNX model (~700 MB^)...
    if not exist "%CACHE_DIR%\model" mkdir "%CACHE_DIR%\model"
    curl -fL --retry 3 -o "%MODEL_PATH%" "%MODEL_URL%"
  )
  set "NER_MODEL_PATH=%MODEL_PATH%"
)

rem ── Locate tokenizer ──────────────────────────────────────────────────────
if not defined NER_TOKENIZER_PATH (
  if exist "%SCRIPT_DIR%tokenizer_export_clean\tokenizer.json" (
    set "NER_TOKENIZER_PATH=%SCRIPT_DIR%tokenizer_export_clean"
    goto :tok_found
  )
  if exist "%SCRIPT_DIR%..\training\multi-head\tokenizer_export_clean\tokenizer.json" (
    set "NER_TOKENIZER_PATH=%SCRIPT_DIR%..\training\multi-head\tokenizer_export_clean"
    goto :tok_found
  )
  echo ERROR: tokenizer_export_clean\ not found next to this script.
  exit /b 1
)
:tok_found

echo Starting NER Demo -^> http://localhost:%NER_PORT%
java -Xms128m -Xmx512m -XX:+UseG1GC -XX:MaxMetaspaceSize=192m ^
  -Dserver.port=%NER_PORT% ^
  "-DNER_MODEL_PATH=%NER_MODEL_PATH%" ^
  "-DNER_TOKENIZER_PATH=%NER_TOKENIZER_PATH%" ^
  -jar "%JAR%"

