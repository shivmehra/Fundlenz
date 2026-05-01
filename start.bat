@echo off
set ROOT=%~dp0

start "Fundlenz Backend" cmd /k "cd /d %ROOT%backend && %ROOT%.venv\Scripts\activate.bat && uvicorn app.main:app --reload --port 8000"
start "Fundlenz Frontend" cmd /k "cd /d %ROOT%frontend && npm run dev"
