#!/usr/bin/env python3
"""
UI Components for Streamlit App

This module contains reusable UI components and styling for the
Fantasy Football Analytics Streamlit application.

Note: Complex UI functions that depend heavily on session state
(like render_keeper_rules_ui, render_import_progress) remain in main.py.
"""

import streamlit as st


def load_custom_css():
    """Load custom CSS styles for the application."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 1rem;
        margin-bottom: 2rem;
        color: white;
    }

    .hero h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .hero p {
        font-size: 1.2rem;
        opacity: 0.95;
        margin-bottom: 2rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-color: #667eea;
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1a202c;
    }

    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }

    .status-queued {
        background: #fef3c7;
        color: #92400e;
    }

    .status-running {
        background: #dbeafe;
        color: #1e40af;
    }

    .status-success {
        background: #d1fae5;
        color: #065f46;
    }

    .status-failed {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Job card */
    .job-card {
        background: linear-gradient(135deg, #f6f8fc 0%, #ffffff 100%);
        border-left: 4px solid #667eea;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .job-id {
        font-family: 'Courier New', monospace;
        background: #f1f5f9;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }

    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .stat-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1.25rem;
        text-align: center;
    }

    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.25rem;
    }

    .stat-label {
        color: #718096;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Timeline */
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.5rem;
        border-left: 2px solid #e2e8f0;
    }

    .timeline-item:last-child {
        border-left: 2px solid transparent;
    }

    .timeline-dot {
        position: absolute;
        left: -0.5rem;
        width: 1rem;
        height: 1rem;
        border-radius: 50%;
        background: #667eea;
        border: 2px solid white;
        box-shadow: 0 0 0 3px #e2e8f0;
    }

    .timeline-content {
        margin-top: -0.25rem;
    }

    /* Button enhancements */
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def render_hero():
    """Render the hero section at the top of the page."""
    st.markdown("""
    <div class="hero">
        <h1>🏈 Fantasy Football Analytics</h1>
        <p>Transform your Yahoo Fantasy Football data into powerful insights</p>
    </div>
    """, unsafe_allow_html=True)


def render_feature_card(icon: str, title: str, description: str):
    """Render a feature card with icon, title, and description."""
    st.markdown(f"""
    <div class="feature-card">
        <div class="feature-icon">{icon}</div>
        <div class="feature-title">{title}</div>
        <div class="feature-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(status: str) -> str:
    """Render a status badge HTML string."""
    status_map = {
        "queued": ("⏳", "status-queued", "Queued"),
        "running": ("🔄", "status-running", "Running"),
        "success": ("✅", "status-success", "Complete"),
        "failed": ("❌", "status-failed", "Failed"),
    }
    icon, css_class, label = status_map.get(status, ("", "status-queued", status))
    return f'<span class="status-badge {css_class}">{icon} {label}</span>'


def render_job_card(job_id: str, league_name: str, status: str):
    """Render a job status card."""
    st.markdown(f"""
    <div class="job-card">
        <h3>🎯 {league_name}</h3>
        <p><strong>Job ID:</strong> <span class="job-id">{job_id}</span></p>
        <p><strong>Status:</strong> {render_status_badge(status)}</p>
    </div>
    """, unsafe_allow_html=True)


def render_timeline():
    """Render a timeline showing the import process steps."""
    st.markdown("""
    <div style="margin: 2rem 0;">
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 1: Connect</strong>
                <p style="color: #718096; font-size: 0.9rem;">Authenticate with your Yahoo account</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 2: Select</strong>
                <p style="color: #718096; font-size: 0.9rem;">Choose your league and season</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 3: Import</strong>
                <p style="color: #718096; font-size: 0.9rem;">Queue your data for processing</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 4: Analyze</strong>
                <p style="color: #718096; font-size: 0.9rem;">Query your data from anywhere</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_stat_card(value: str, label: str):
    """Render a stat card with value and label."""
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)
