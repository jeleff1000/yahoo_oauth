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
    """Load custom CSS styles for the application (supports light/dark mode and mobile)."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* CSS Variables for theming - adapts to Streamlit's theme */
    :root {
        --card-bg: rgba(255, 255, 255, 0.95);
        --card-border: rgba(0, 0, 0, 0.1);
        --text-primary: inherit;
        --text-secondary: rgba(0, 0, 0, 0.6);
        --accent-color: #667eea;
        --accent-light: rgba(102, 126, 234, 0.1);
    }

    /* Dark mode overrides - detect via Streamlit's data-theme or prefers-color-scheme */
    @media (prefers-color-scheme: dark) {
        :root {
            --card-bg: rgba(30, 30, 30, 0.95);
            --card-border: rgba(255, 255, 255, 0.1);
            --text-primary: inherit;
            --text-secondary: rgba(255, 255, 255, 0.6);
            --accent-light: rgba(102, 126, 234, 0.2);
        }
    }

    /* Streamlit dark theme detection */
    [data-testid="stAppViewContainer"][data-theme="dark"],
    .stApp[data-theme="dark"] {
        --card-bg: rgba(30, 30, 30, 0.95);
        --card-border: rgba(255, 255, 255, 0.1);
        --text-secondary: rgba(255, 255, 255, 0.6);
        --accent-light: rgba(102, 126, 234, 0.2);
    }

    /* Global styles */
    .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero section */
    .hero {
        text-align: center;
        padding: 0.75rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        color: white;
    }

    .hero h1 {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.2;
    }

    .hero p {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* Feature cards - theme aware */
    .feature-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin-bottom: 0.75rem;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-color: var(--accent-color);
    }

    .feature-icon {
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }

    .feature-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }

    .feature-desc {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.4;
    }

    /* Status badge - consistent colors for visibility */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
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

    /* Job card - theme aware */
    .job-card {
        background: var(--card-bg);
        border-left: 4px solid var(--accent-color);
        border-radius: 0.5rem;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .job-card h3 {
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }

    .job-card p {
        color: var(--text-secondary);
        margin-bottom: 0.5rem;
    }

    .job-id {
        font-family: 'Courier New', monospace;
        background: var(--accent-light);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }

    /* Stats grid - responsive */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 1rem 0;
    }

    .stat-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }

    .stat-value {
        font-size: clamp(1.5rem, 4vw, 2rem);
        font-weight: 700;
        color: var(--accent-color);
        margin-bottom: 0.25rem;
    }

    .stat-label {
        color: var(--text-secondary);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Timeline - theme aware */
    .timeline-item {
        position: relative;
        padding-left: 2rem;
        padding-bottom: 1.25rem;
        border-left: 2px solid var(--card-border);
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
        background: var(--accent-color);
        border: 2px solid var(--card-bg);
        box-shadow: 0 0 0 2px var(--card-border);
    }

    .timeline-content {
        margin-top: -0.25rem;
    }

    .timeline-content strong {
        color: var(--text-primary);
    }

    .timeline-content p {
        color: var(--text-secondary);
        font-size: 0.85rem;
        margin-top: 0.25rem;
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

    /* Mobile-specific adjustments */
    @media (max-width: 768px) {
        .hero {
            padding: 1rem 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 0.75rem;
        }

        .hero h1 {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }

        .hero p {
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }

        .feature-card {
            padding: 1rem;
        }

        .feature-icon {
            font-size: 1.5rem;
        }

        .job-card {
            padding: 1rem;
        }

        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .timeline-item {
            padding-left: 1.5rem;
        }
    }

    /* Extra small screens */
    @media (max-width: 480px) {
        .stats-grid {
            grid-template-columns: 1fr;
        }

        .status-badge {
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
        }
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
        <p>Import or explore fantasy leagues from any season</p>
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
    <div style="margin: 1.5rem 0;">
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 1: Connect</strong>
                <p>Authenticate with your Yahoo account</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 2: Select</strong>
                <p>Choose your league and season</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 3: Import</strong>
                <p>Queue your data for processing</p>
            </div>
        </div>
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>Step 4: Analyze</strong>
                <p>Query your data from anywhere</p>
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
