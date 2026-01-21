#!/bin/bash
# Quick launch script for CVSense

echo "🚀 CVSense - Intelligent Resume Screening"
echo "=========================================="
echo ""
echo "Launching automated upload interface..."
echo "Open your browser to: http://localhost:8501"
echo ""
echo "Features:"
echo "  ✅ Upload resumes (PDF/TXT) in batch"
echo "  ✅ Add job descriptions via text or file"
echo "  ✅ Get instant matching results"
echo "  ✅ Export rankings to CSV"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py
