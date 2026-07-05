@echo off
echo ========================================================
echo Starting Xoodrip Content Intelligence with CUDA support!
echo Using Virtual Environment: D:\xoodrip-venv
echo ========================================================

D:\xoodrip-venv\Scripts\python.exe -m uvicorn app.main:app --reload
