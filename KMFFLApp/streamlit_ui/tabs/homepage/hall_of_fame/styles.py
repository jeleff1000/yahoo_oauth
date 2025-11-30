# tabs/hall_of_fame/styles.py
"""
Hall of Fame styles - Simplified and integrated with design tokens.

Uses the unified theme system with static cards (no hover effects on non-interactive elements).
"""
import streamlit as st


def apply_hall_of_fame_styles():
    """Apply clean CSS styles for Hall of Fame sections."""
    st.markdown("""
        <style>
        /* ========================================
           Hall of Fame Hero Banner - Subtle gradient
           ======================================== */
        .hof-hero {
            background: linear-gradient(135deg,
                var(--gradient-start, rgba(102, 126, 234, 0.1)) 0%,
                var(--gradient-end, rgba(118, 75, 162, 0.06)) 100%);
            padding: var(--space-xl, 2rem);
            border-radius: var(--radius-lg, 12px);
            margin-bottom: var(--space-lg, 1.5rem);
            border: 1px solid var(--border, #E5E7EB);
            text-align: center;
        }
        .hof-hero h1 {
            color: var(--text-primary, #1F2937) !important;
            font-size: 2rem;
            margin: 0;
            font-weight: 700;
        }
        .hof-hero p {
            color: var(--text-secondary, #6B7280) !important;
            font-size: 1.1rem;
            margin: 0.75rem 0 0 0;
        }

        /* ========================================
           Section Headers - Clean accent underline
           ======================================== */
        .hof-section-header {
            margin: var(--space-lg, 1.5rem) 0 var(--space-md, 1rem) 0;
            padding-bottom: var(--space-sm, 0.5rem);
            border-bottom: 2px solid var(--accent, #667eea);
        }
        .hof-section-header h2 {
            margin: 0;
            font-size: 1.5rem;
            color: var(--text-primary, #1F2937) !important;
            font-weight: 600;
        }
        .hof-section-header p {
            margin: 0.25rem 0 0 0;
            color: var(--text-secondary, #6B7280) !important;
        }

        /* ========================================
           Season Cards - Static display
           ======================================== */
        .hof-season-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            border-radius: var(--radius-md, 8px);
            padding: var(--space-md, 1rem);
            margin-bottom: var(--space-sm, 0.5rem);
            /* Static - no shadow, no hover */
        }
        .hof-season-card .season-rank {
            font-size: 1.1rem;
            font-weight: bold;
            color: var(--accent, #667eea);
        }
        .hof-season-card .season-year {
            color: var(--text-muted, #9CA3AF);
            font-weight: 600;
        }
        .hof-season-card .season-manager {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-season-card .season-stat-label {
            font-size: 0.8rem;
            color: var(--text-muted, #9CA3AF);
        }
        .hof-season-card .season-stat-value {
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-season-card .stat-highlight {
            color: var(--success, #10B981);
        }

        /* ========================================
           Game Cards - Static display
           ======================================== */
        .hof-game-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            border-radius: var(--radius-md, 8px);
            padding: var(--space-sm, 0.5rem) var(--space-md, 1rem);
            margin-bottom: var(--space-sm, 0.5rem);
        }
        .hof-game-card-playoff {
            border-left: 3px solid var(--warning, #F59E0B);
        }
        .hof-game-card .game-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: var(--space-xs, 0.25rem);
        }
        .hof-game-card .game-date {
            color: var(--text-muted, #9CA3AF);
            font-size: 0.85rem;
        }
        .hof-game-card .game-stat {
            font-weight: 600;
            color: var(--success, #10B981);
        }
        .hof-game-card .team-name {
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-game-card .team-score {
            font-weight: 700;
            color: var(--text-primary, #1F2937);
        }
        .hof-game-card .loser-name,
        .hof-game-card .loser-score {
            color: var(--text-muted, #9CA3AF);
        }

        /* ========================================
           Timeline Cards - Static display
           ======================================== */
        .hof-timeline-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            border-left: 3px solid var(--accent, #667eea);
            border-radius: var(--radius-md, 8px);
            padding: var(--space-md, 1rem);
            margin-bottom: var(--space-sm, 0.5rem);
        }
        .hof-timeline-card .timeline-year {
            font-weight: 600;
            color: var(--accent, #667eea);
            font-size: 0.9rem;
        }
        .hof-timeline-card .timeline-winner {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-timeline-card .timeline-details {
            font-size: 0.85rem;
            color: var(--text-secondary, #6B7280);
        }

        /* ========================================
           Week Cards - Static display
           ======================================== */
        .hof-week-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            padding: var(--space-sm, 0.5rem) var(--space-md, 1rem);
            border-radius: var(--radius-sm, 4px);
            margin-bottom: var(--space-xs, 0.25rem);
        }
        .hof-week-card-playoff {
            border-left: 3px solid var(--warning, #F59E0B);
        }
        .hof-week-card .week-manager {
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-week-card .week-meta {
            font-size: 0.85rem;
            color: var(--text-muted, #9CA3AF);
        }
        .hof-week-card .week-score {
            font-weight: 700;
            color: var(--text-primary, #1F2937);
        }

        /* ========================================
           Leader Cards - Static display
           ======================================== */
        .hof-leader-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            border-radius: var(--radius-md, 8px);
            padding: var(--space-md, 1rem);
            text-align: center;
        }
        .hof-leader-card .leader-medal {
            font-size: 2rem;
            margin-bottom: var(--space-xs, 0.25rem);
        }
        .hof-leader-card .leader-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary, #1F2937);
        }
        .hof-leader-card .leader-score {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--success, #10B981);
        }
        .hof-leader-card .leader-label {
            font-size: 0.8rem;
            color: var(--text-muted, #9CA3AF);
        }

        /* ========================================
           Record Cards - Static display
           ======================================== */
        .hof-record-card {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            padding: var(--space-md, 1rem);
            border-radius: var(--radius-md, 8px);
            margin: var(--space-sm, 0.5rem) 0;
        }
        .hof-record-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-muted, #9CA3AF);
        }
        .hof-record-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary, #1F2937);
        }
        .hof-record-detail {
            font-size: 0.9rem;
            color: var(--text-secondary, #6B7280);
        }

        /* ========================================
           Championship Badges - Keep some color
           ======================================== */
        .hof-champ-badge {
            padding: var(--space-md, 1rem);
            border-radius: var(--radius-md, 8px);
            text-align: center;
            margin-bottom: var(--space-sm, 0.5rem);
        }
        .hof-champ-badge .badge-icon { font-size: 1.5rem; }
        .hof-champ-badge .badge-name { font-weight: 600; }
        .hof-champ-badge .badge-count { font-size: 1.2rem; font-weight: 700; }
        .hof-champ-badge .badge-label { font-size: 0.75rem; }

        .hof-champ-badge-gold {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: #451a03;
        }
        .hof-champ-badge-silver {
            background: linear-gradient(135deg, #9ca3af, #d1d5db);
            color: #1f2937;
        }
        .hof-champ-badge-bronze {
            background: linear-gradient(135deg, #d97706, #ea580c);
            color: white;
        }

        /* ========================================
           Leaderboard Rows - Static display
           ======================================== */
        .hof-leaderboard-row {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            display: flex;
            align-items: center;
            padding: var(--space-md, 1rem);
            border-radius: var(--radius-md, 8px);
            margin: var(--space-xs, 0.25rem) 0;
            color: var(--text-primary, #1F2937);
        }
        .hof-leaderboard-row:nth-child(1) {
            border-left: 3px solid #fbbf24;
        }
        .hof-leaderboard-row:nth-child(2) {
            border-left: 3px solid #9ca3af;
        }
        .hof-leaderboard-row:nth-child(3) {
            border-left: 3px solid #d97706;
        }

        /* ========================================
           Stats Grid - Static display
           ======================================== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: var(--space-md, 1rem);
            margin: var(--space-md, 1rem) 0;
        }
        .stat-box {
            background: var(--bg-secondary, #F8F9FA);
            border: 1px solid var(--border, #E5E7EB);
            padding: var(--space-md, 1rem);
            border-radius: var(--radius-md, 8px);
            text-align: center;
        }
        .stat-box h4 {
            margin: 0;
            font-size: 0.8rem;
            color: var(--text-muted, #9CA3AF);
            text-transform: uppercase;
        }
        .stat-box .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--accent, #667eea);
        }
        .stat-box .detail {
            font-size: 0.85rem;
            color: var(--text-secondary, #6B7280);
        }

        /* ========================================
           Badges
           ======================================== */
        .badge {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: var(--radius-full, 20px);
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        .badge-gold {
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: #451a03;
        }
        .badge-silver {
            background: linear-gradient(135deg, #9ca3af, #d1d5db);
            color: #1f2937;
        }
        .badge-bronze {
            background: linear-gradient(135deg, #d97706, #ea580c);
            color: white;
        }

        /* ========================================
           Responsive Design
           ======================================== */
        @media (max-width: 768px) {
            .hof-hero h1 { font-size: 1.5rem; }
            .hof-hero p { font-size: 0.95rem; }
            .hof-hero { padding: var(--space-md, 1rem); }

            .stats-grid { grid-template-columns: repeat(2, 1fr); }

            .hof-section-header h2 { font-size: 1.25rem; }

            .hof-game-card,
            .hof-season-card,
            .hof-timeline-card {
                padding: var(--space-sm, 0.5rem);
            }

            .stat-box .value { font-size: 1.5rem; }
        }

        @media (max-width: 480px) {
            .hof-hero h1 { font-size: 1.3rem; }
            .stats-grid { grid-template-columns: 1fr; }

            .hof-game-card .game-header {
                flex-direction: column;
                gap: 0.15rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)
