@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   ModTranslator
echo ============================================================
echo.

if exist venv\Scripts\activate.bat (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
    echo Iniciando GUI...
    modtranslator-gui
    goto :eof
)

:: ── Primera vez ──────────────────────────────────────────────
echo Primera ejecucion detectada. Configurando entorno...
echo.

echo [1/4] Comprobando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Python no encontrado.
    echo.
    echo Instala Python 3.10 o superior desde:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANTE: marca "Add Python to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo ERROR: No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo OK
echo.

echo [3/4] Activando entorno virtual...
call venv\Scripts\activate.bat
echo OK
echo.

echo [4/4] Instalando dependencias base...
echo.

:: Escribir script Python para mostrar spinner durante pip install
echo import subprocess,sys > _mt_install.py
echo chars='^|/-\\' >> _mt_install.py
echo proc=subprocess.Popen([sys.executable,'-u','-m','pip','install','-e','.[gui]'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True) >> _mt_install.py
echo i=0 >> _mt_install.py
echo for line in proc.stdout: >> _mt_install.py
echo     line=line.strip() >> _mt_install.py
echo     if line: >> _mt_install.py
echo         print('\r  '+chars[i%%4]+' '+line[:65].ljust(65),end='',flush=True) >> _mt_install.py
echo         i+=1 >> _mt_install.py
echo proc.wait() >> _mt_install.py
echo print('\r  OK'.ljust(70)) >> _mt_install.py
echo sys.exit(proc.returncode) >> _mt_install.py

python _mt_install.py
set INST_ERR=%errorlevel%
del _mt_install.py 2>nul

if %INST_ERR% neq 0 (
    echo.
    echo ERROR: Fallo al instalar dependencias.
    echo Comprueba tu conexion a internet e intentalo de nuevo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Listo. Iniciando GUI...
echo   (La primera vez se instalan los modelos de IA segun tu GPU.
echo    Puedes verlo en el log de la aplicacion.)
echo ============================================================
echo.
modtranslator-gui
