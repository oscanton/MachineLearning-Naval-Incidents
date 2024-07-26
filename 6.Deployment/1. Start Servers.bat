@echo off
echo Activate base environment of Anaconda and start local servers?
pause

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base

start "First server (Incident involment)" python server_FastAPI.py
start "Second server (Incident type)" python server_FastAPI_MA.py

