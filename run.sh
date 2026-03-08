#!/bin/bash
trap 'kill 0' EXIT
streamlit run dashboard/app.py &
uvicorn api.main:app --reload &
wait
