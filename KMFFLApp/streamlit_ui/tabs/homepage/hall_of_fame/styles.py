# tabs/hall_of_fame/styles.py
import streamlit as st


def apply_hall_of_fame_styles():
    """Apply custom CSS styles for Hall of Fame sections - universal colors for light/dark mode"""
    st.markdown("""
        <style>
        /* ========================================
           FIXED COLOR SCHEME
           Colors that work in BOTH light and dark mode
           Using darker backgrounds with light text
           ======================================== */

        /* ========================================
           Hall of Fame Hero Banner
           ======================================== */
        .hof-hero {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2.5rem 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
            text-align: center;
        }

        .hof-hero h1 {
            color: white !important;
            font-size: 3rem;
            margin: 0;
            font-weight: 800;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }

        .hof-hero p {
            color: rgba(255,255,255,0.95) !important;
            font-size: 1.3rem;
            margin: 1rem 0 0 0;
        }

        /* ========================================
           Streamlit Native Metric Cards (KPIs)
           Styles st.metric() components
           ======================================== */
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%) !important;
            border-radius: 10px !important;
            padding: 1rem !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2) !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
        }

        [data-testid="stMetricLabel"] {
            color: #93c5fd !important;
        }

        [data-testid="stMetricLabel"] p {
            color: #93c5fd !important;
            font-weight: 600 !important;
        }

        [data-testid="stMetricValue"] {
            color: white !important;
        }

        [data-testid="stMetricValue"] div {
            color: white !important;
        }

        [data-testid="stMetricDelta"] {
            color: #4ade80 !important;
        }

        [data-testid="stMetricDelta"] svg {
            display: none !important;
        }

        [data-testid="stMetricDelta"] div {
            color: #93c5fd !important;
            font-size: 0.85rem !important;
        }

        /* ========================================
           Streamlit Expander Styling
           ======================================== */
        [data-testid="stExpander"] {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(100, 116, 139, 0.3) !important;
        }

        [data-testid="stExpander"] summary {
            color: white !important;
        }

        [data-testid="stExpander"] summary span {
            color: white !important;
        }

        [data-testid="stExpander"] svg {
            color: #94a3b8 !important;
            fill: #94a3b8 !important;
        }

        /* Expander content area */
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background: transparent !important;
        }

        /* ========================================
           Streamlit Selectbox/Multiselect Styling
           ======================================== */
        [data-testid="stSelectbox"],
        [data-testid="stMultiSelect"] {
            background: transparent !important;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label {
            color: #e2e8f0 !important;
        }

        /* ========================================
           Streamlit Checkbox/Toggle Styling
           ======================================== */
        [data-testid="stCheckbox"] label span {
            color: #e2e8f0 !important;
        }

        /* ========================================
           Streamlit Info/Warning Boxes
           ======================================== */
        [data-testid="stAlert"] {
            background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%) !important;
            border-radius: 8px !important;
            color: white !important;
        }

        [data-testid="stAlert"] p {
            color: white !important;
        }

        /* ========================================
           Season Cards - Slate Blue
           ======================================== */
        .hof-season-card {
            background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
            border-left: 4px solid #667eea;
        }

        .hof-season-card:hover {
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
            transform: translateY(-2px);
        }

        .hof-season-card .season-rank {
            font-size: 1.2rem;
            font-weight: bold;
            color: #fbbf24 !important;
        }

        .hof-season-card .season-year {
            color: #a5b4fc !important;
            font-weight: bold;
        }

        .hof-season-card .season-manager {
            font-size: 1.1rem;
            font-weight: bold;
            color: white !important;
        }

        .hof-season-card .season-stat-label {
            font-size: 0.8rem;
            color: #94a3b8 !important;
        }

        .hof-season-card .season-stat-value {
            font-size: 1.1rem;
            font-weight: bold;
            color: white !important;
        }

        .hof-season-card .stat-highlight {
            color: #4ade80 !important;
        }

        /* ========================================
           Game Cards - Dark Purple
           ======================================== */
        .hof-game-card {
            background: linear-gradient(135deg, #3b2667 0%, #4a3674 100%) !important;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease;
        }

        .hof-game-card:hover {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transform: translateY(-1px);
        }

        .hof-game-card-playoff {
            border-left: 3px solid #fbbf24 !important;
            background: linear-gradient(135deg, #4a3520 0%, #5c4428 100%) !important;
        }

        .hof-game-card .game-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.4rem;
        }

        .hof-game-card .game-date {
            color: #a5b4fc !important;
            font-weight: bold;
            font-size: 0.85rem;
        }

        .hof-game-card .game-stat {
            font-weight: bold;
            color: #4ade80 !important;
        }

        .hof-game-card .team-name {
            font-weight: 600;
            color: white !important;
        }

        .hof-game-card .team-score {
            font-weight: bold;
            color: white !important;
        }

        .hof-game-card .loser-name {
            color: #94a3b8 !important;
        }

        .hof-game-card .loser-score {
            color: #94a3b8 !important;
        }

        /* ========================================
           Timeline Cards - Indigo
           ======================================== */
        .hof-timeline-card {
            background: linear-gradient(135deg, #312e81 0%, #4338ca 100%) !important;
            border-left: 4px solid #a5b4fc;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .hof-timeline-card .timeline-year {
            font-weight: bold;
            color: #fbbf24 !important;
            font-size: 0.9rem;
            margin-bottom: 0.4rem;
        }

        .hof-timeline-card .timeline-winner {
            font-size: 1.1rem;
            font-weight: bold;
            color: white !important;
            margin-bottom: 0.3rem;
        }

        .hof-timeline-card .timeline-details {
            font-size: 0.85rem;
            color: #c7d2fe !important;
        }

        /* ========================================
           Week Cards - Teal
           ======================================== */
        .hof-week-card {
            background: linear-gradient(135deg, #134e4a 0%, #115e59 100%) !important;
            padding: 0.6rem 0.8rem;
            border-radius: 6px;
            margin-bottom: 0.4rem;
        }

        .hof-week-card-playoff {
            background: linear-gradient(135deg, #4a3520 0%, #5c4428 100%) !important;
            border-left: 3px solid #fbbf24;
        }

        .hof-week-card .week-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .hof-week-card .week-manager {
            font-weight: bold;
            color: white !important;
        }

        .hof-week-card .week-meta {
            font-size: 0.85rem;
            color: #5eead4 !important;
        }

        .hof-week-card .week-score {
            font-weight: bold;
            font-size: 1.1rem;
            color: white !important;
        }

        /* ========================================
           Leader Cards - Deep Purple
           ======================================== */
        .hof-leader-card {
            background: linear-gradient(135deg, #4c1d95 0%, #6d28d9 100%) !important;
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(109, 40, 217, 0.3);
        }

        .hof-leader-card .leader-medal {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }

        .hof-leader-card .leader-name {
            font-size: 1.3rem;
            font-weight: bold;
            color: white !important;
            margin-bottom: 0.8rem;
        }

        .hof-leader-card .leader-score {
            font-size: 1.8rem;
            font-weight: bold;
            color: #4ade80 !important;
            margin-bottom: 0.5rem;
        }

        .hof-leader-card .leader-label {
            font-size: 0.85rem;
            color: #c4b5fd !important;
            margin-bottom: 0.8rem;
        }

        .hof-leader-card .leader-stats {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
            text-align: center;
        }

        .hof-leader-card .stat-label {
            font-size: 0.75rem;
            color: #c4b5fd !important;
        }

        .hof-leader-card .stat-value {
            font-weight: 600;
            color: white !important;
        }

        /* ========================================
           Rivalry Cards - Dark Cyan
           ======================================== */
        .hof-rivalry-card {
            background: linear-gradient(135deg, #164e63 0%, #0e7490 100%) !important;
            border-radius: 8px;
            padding: 0.8rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }

        .hof-rivalry-card .rivalry-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .hof-rivalry-card .rivalry-teams {
            font-weight: bold;
            color: white !important;
        }

        .hof-rivalry-card .rivalry-games {
            color: #67e8f9 !important;
            font-weight: bold;
        }

        .hof-rivalry-card .rivalry-stats {
            display: flex;
            justify-content: space-between;
            font-size: 0.9rem;
            color: #a5f3fc !important;
        }

        /* ========================================
           Section Headers - Slate
           ======================================== */
        .hof-section-header {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important;
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            margin: 1.5rem 0;
        }

        .hof-section-header h2 {
            margin: 0;
            font-size: 1.8rem;
            color: white !important;
        }

        .hof-section-header p {
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
            color: #94a3b8 !important;
        }

        /* ========================================
           Records Header - Purple Border
           ======================================== */
        .hof-records-header {
            background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%) !important;
            padding: 1.5rem;
            border-radius: 12px;
            border: 2px solid #a855f7;
            margin-bottom: 1.5rem;
        }

        .hof-records-header h2 {
            margin: 0;
            color: #e879f9 !important;
        }

        .hof-records-header p {
            margin: 0.5rem 0 0 0;
            color: #c4b5fd !important;
        }

        /* ========================================
           Bracket Header - Green Accent
           ======================================== */
        .hof-bracket-header {
            background: linear-gradient(135deg, #14532d 0%, #166534 100%) !important;
            padding: 2rem;
            border-radius: 16px;
            border-left: 6px solid #4ade80;
            margin-bottom: 2rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        .hof-bracket-header h2 {
            margin: 0;
            color: #4ade80 !important;
            font-size: 2rem;
        }

        .hof-bracket-header p {
            margin: 0.75rem 0 0 0;
            font-size: 1.05rem;
            color: #86efac !important;
        }

        /* ========================================
           Record Cards - Dark Blue
           ======================================== */
        .hof-record-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%) !important;
            padding: 1.25rem;
            border-radius: 12px;
            border: 2px solid #3b82f6;
            margin: 0.75rem 0;
            transition: all 0.3s ease;
        }

        .hof-record-card:hover {
            border-color: #60a5fa;
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.3);
            transform: translateY(-2px);
        }

        .hof-record-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.25rem;
            color: #93c5fd !important;
        }

        .hof-record-value {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            color: white !important;
        }

        .hof-record-detail {
            font-size: 1rem;
            margin-top: 0.25rem;
            color: #bfdbfe !important;
        }

        /* ========================================
           Leaderboard Rows - Slate
           ======================================== */
        .hof-leaderboard-row {
            background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
            display: flex;
            align-items: center;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 4px solid transparent;
            transition: all 0.2s ease;
            color: white !important;
        }

        .hof-leaderboard-row:nth-child(1) {
            border-left-color: #fbbf24;
            background: linear-gradient(135deg, #4a3520 0%, #5c4428 100%) !important;
        }

        .hof-leaderboard-row:nth-child(2) {
            border-left-color: #9ca3af;
            background: linear-gradient(135deg, #374151 0%, #4b5563 100%) !important;
        }

        .hof-leaderboard-row:nth-child(3) {
            border-left-color: #d97706;
            background: linear-gradient(135deg, #451a03 0%, #78350f 100%) !important;
        }

        .hof-leaderboard-row:hover {
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            transform: translateX(5px);
        }

        /* ========================================
           Championship Badges - Keep Vibrant
           ======================================== */
        .hof-champ-badge {
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 0.5rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2);
        }

        .hof-champ-badge .badge-icon {
            font-size: 1.8rem;
        }

        .hof-champ-badge .badge-name {
            font-weight: bold;
            font-size: 1.1rem;
            margin: 0.3rem 0;
        }

        .hof-champ-badge .badge-count {
            font-size: 1.3rem;
            font-weight: bold;
        }

        .hof-champ-badge .badge-label {
            font-size: 0.8rem;
            margin-top: 0.2rem;
        }

        /* Gold badge - 3+ championships */
        .hof-champ-badge-gold {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
        }
        .hof-champ-badge-gold .badge-name,
        .hof-champ-badge-gold .badge-count {
            color: #451a03 !important;
        }
        .hof-champ-badge-gold .badge-label {
            color: #78350f !important;
        }

        /* Silver badge - 2 championships */
        .hof-champ-badge-silver {
            background: linear-gradient(135deg, #9ca3af 0%, #d1d5db 100%) !important;
        }
        .hof-champ-badge-silver .badge-name,
        .hof-champ-badge-silver .badge-count {
            color: #1f2937 !important;
        }
        .hof-champ-badge-silver .badge-label {
            color: #374151 !important;
        }

        /* Bronze badge - 1 championship */
        .hof-champ-badge-bronze {
            background: linear-gradient(135deg, #d97706 0%, #ea580c 100%) !important;
        }
        .hof-champ-badge-bronze .badge-name,
        .hof-champ-badge-bronze .badge-count,
        .hof-champ-badge-bronze .badge-label {
            color: white !important;
        }

        /* ========================================
           Gradient Headers - Keep Vibrant
           ======================================== */
        .hof-gradient-header {
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        }

        .hof-gradient-header h2,
        .hof-gradient-header h3,
        .hof-gradient-header h4 {
            margin: 0;
            color: white !important;
            font-size: 1.8rem;
        }

        .hof-gradient-header p {
            margin: 0.5rem 0 0 0;
            color: rgba(255,255,255,0.9) !important;
            font-size: 1rem;
        }

        .hof-header-gold {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
            box-shadow: 0 4px 20px rgba(251, 191, 36, 0.3);
        }
        .hof-header-gold h2, .hof-header-gold h3, .hof-header-gold h4 {
            color: #451a03 !important;
        }
        .hof-header-gold p {
            color: #78350f !important;
        }

        .hof-header-purple {
            background: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%) !important;
            box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3);
        }

        .hof-header-red {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
            box-shadow: 0 4px 16px rgba(220, 38, 38, 0.3);
        }

        .hof-header-orange {
            background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
            box-shadow: 0 4px 16px rgba(234, 88, 12, 0.3);
        }

        .hof-header-green {
            background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%) !important;
            box-shadow: 0 4px 20px rgba(22, 163, 74, 0.3);
        }

        .hof-header-violet {
            background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%) !important;
            box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        }

        .hof-header-fire {
            background: linear-gradient(135deg, #dc2626 0%, #ea580c 100%) !important;
            box-shadow: 0 4px 20px rgba(220, 38, 38, 0.3);
        }

        .hof-header-blue {
            background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%) !important;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
        }

        .hof-header-teal {
            background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
            box-shadow: 0 4px 20px rgba(13, 148, 136, 0.3);
        }

        /* ========================================
           Stats Grid
           ======================================== */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            color: white !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }

        .stat-box h4 {
            margin: 0;
            font-size: 0.9rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: white !important;
        }

        .stat-box .value {
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0.5rem 0;
            color: white !important;
        }

        .stat-box .detail {
            font-size: 0.9rem;
            opacity: 0.85;
            color: white !important;
        }

        /* ========================================
           Badge Styles
           ======================================== */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.25rem;
        }

        .badge-gold {
            background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
            color: #451a03 !important;
        }

        .badge-silver {
            background: linear-gradient(135deg, #9ca3af, #d1d5db) !important;
            color: #1f2937 !important;
        }

        .badge-bronze {
            background: linear-gradient(135deg, #d97706, #ea580c) !important;
            color: white !important;
        }

        .badge-dynasty {
            background: linear-gradient(135deg, #dc2626, #ea580c) !important;
            color: white !important;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
        }

        /* ========================================
           Trophy Icon
           ======================================== */
        .trophy-icon {
            font-size: 3rem;
            display: inline-block;
            margin-right: 0.5rem;
        }

        /* ========================================
           Champion Card
           ======================================== */
        .champion-card {
            background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%) !important;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 4px 15px rgba(251, 191, 36, 0.3);
            border: 2px solid #d97706;
        }

        .champion-card h3 {
            margin: 0 0 0.5rem 0;
            color: #451a03 !important;
            font-size: 1.5rem;
        }

        /* ========================================
           Tab Styling
           ======================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(51, 65, 85, 0.5);
            padding: 0.5rem;
            border-radius: 12px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea, #764ba2) !important;
            color: white !important;
        }

        /* ========================================
           Responsive Design
           ======================================== */
        @media (max-width: 768px) {
            .hof-hero h1 {
                font-size: 1.8rem;
            }

            .hof-hero p {
                font-size: 0.95rem;
            }

            .hof-hero {
                padding: 1.5rem 1rem;
            }

            .stats-grid {
                grid-template-columns: 1fr;
            }

            .hof-gradient-header {
                padding: 1rem;
            }

            .hof-gradient-header h2,
            .hof-gradient-header h3,
            .hof-gradient-header h4 {
                font-size: 1.4rem;
            }

            .hof-gradient-header p {
                font-size: 0.9rem;
            }

            .hof-section-header h2 {
                font-size: 1.4rem;
            }

            .hof-game-card,
            .hof-season-card,
            .hof-timeline-card {
                padding: 0.6rem;
            }

            .hof-champ-badge {
                padding: 0.75rem;
            }

            .hof-champ-badge .badge-icon {
                font-size: 1.4rem;
            }

            .hof-champ-badge .badge-name {
                font-size: 0.95rem;
            }

            .hof-champ-badge .badge-count {
                font-size: 1.1rem;
            }

            .stat-box {
                padding: 1rem;
            }

            .stat-box .value {
                font-size: 1.8rem;
            }

            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem 0.75rem;
                font-size: 0.85rem;
            }
        }

        @media (max-width: 480px) {
            .hof-hero h1 {
                font-size: 1.5rem;
            }

            .hof-game-card .game-header {
                flex-direction: column;
                gap: 0.2rem;
            }

            .hof-rivalry-card .rivalry-stats {
                flex-direction: column;
                gap: 0.2rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)
