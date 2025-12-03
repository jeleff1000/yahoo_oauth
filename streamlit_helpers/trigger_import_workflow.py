#!/usr/bin/env python3
"""
Helper script to trigger the GitHub Actions workflow from Streamlit

This script is called by the Streamlit app when a user wants to create
their own fantasy football analytics site.
"""

import os
import json
import requests
from typing import Dict, Optional
from datetime import datetime, timezone
import hashlib


def generate_user_id(league_id: str, season: int) -> str:
    """Generate a unique user ID based on league and season"""
    input_str = f"{league_id}_{season}_{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha256(input_str.encode()).hexdigest()[:16]


def trigger_import_workflow(
    league_data: Dict,
    github_token: str,
    repo_owner: str = "jeleff1000",
    repo_name: str = "yahoo_oauth",
) -> Dict:
    """
    Trigger the league import workflow via GitHub Actions API

    Args:
        league_data: Dictionary containing:
            - league_id: Yahoo league ID
            - league_name: Display name of the league
            - season: Year (end_year)
            - start_year: First season to import (optional, defaults to season)
            - oauth_token: Yahoo OAuth credentials
            - num_teams: Number of teams (optional, default 10)
            - playoff_teams: Number of playoff teams (optional, default 6)
            - regular_season_weeks: Number of regular season weeks (optional, default 14)
            - manager_name_overrides: Dict of name overrides (optional)
        github_token: GitHub personal access token with repo/workflow permissions
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name

    Returns:
        Dictionary with:
            - success: bool
            - user_id: Unique identifier for this import job
            - workflow_run_url: URL to track the workflow
            - message: Status message
    """
    import base64

    # Validate required fields
    required_fields = ['league_id', 'league_name', 'season', 'oauth_token']
    missing = [f for f in required_fields if f not in league_data]
    if missing:
        return {
            'success': False,
            'error': f"Missing required fields: {', '.join(missing)}"
        }

    # Generate unique user ID
    user_id = generate_user_id(league_data['league_id'], league_data['season'])

    # Base64 encode the JSON to avoid shell escaping issues in GitHub Actions
    league_data_json = json.dumps(league_data)
    league_data_b64 = base64.b64encode(league_data_json.encode()).decode()

    # Debug: Log what we're sending (helps troubleshoot truncation issues)
    print(f"[DEBUG] Sending league_id: {league_data.get('league_id')}")
    print(f"[DEBUG] Sending league_name: {league_data.get('league_name')}")
    print(f"[DEBUG] Sending season: {league_data.get('season')}")
    print(f"[DEBUG] Sending start_year: {league_data.get('start_year')}")
    print(f"[DEBUG] Base64 length: {len(league_data_b64)}")

    # Prepare workflow inputs
    workflow_inputs = {
        'league_data_b64': league_data_b64,
        'user_id': user_id
    }

    # GitHub API endpoint
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/league_import_worker.yml/dispatches"

    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}',
        'Content-Type': 'application/json'
    }

    payload = {
        'ref': 'main',  # or your default branch
        'inputs': workflow_inputs
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 204:
            # Workflow triggered successfully
            # Get the run URL (may need to query runs API)
            workflow_run_url = f"https://github.com/{repo_owner}/{repo_name}/actions/workflows/league_import_worker.yml"

            return {
                'success': True,
                'user_id': user_id,
                'workflow_run_url': workflow_run_url,
                'message': f"Import started for {league_data['league_name']} ({league_data['season']})",
                'estimated_time': '60-120 minutes'
            }
        else:
            return {
                'success': False,
                'error': f"GitHub API error: {response.status_code}",
                'details': response.text
            }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Network error: {str(e)}"
        }


def get_workflow_run_id_from_motherduck(
    user_id: str,
    max_attempts: int = 10,
) -> Optional[int]:
    """
    Get workflow run ID from MotherDuck job tracking table.

    This is the preferred method as it uses the user_id to find the exact run,
    avoiding race conditions when multiple users trigger workflows simultaneously.

    Args:
        user_id: The unique user ID passed to the workflow
        max_attempts: Number of times to poll before giving up

    Returns:
        Run ID if found, None otherwise
    """
    import time
    import os

    try:
        import duckdb
    except ImportError:
        return None

    # Try to get token from environment or streamlit secrets
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        try:
            import streamlit as st
            token = st.secrets.get("MOTHERDUCK_TOKEN")
        except:
            pass

    if not token:
        return None

    for attempt in range(max_attempts):
        try:
            # Use connection string with token to avoid global env mutation
            con = duckdb.connect(f"md:?motherduck_token={token}")
            result = con.execute("""
                SELECT workflow_run_id
                FROM ops.import_jobs
                WHERE user_id = ?
            """, [user_id]).fetchone()
            con.close()

            if result and result[0]:
                return int(result[0])

            # Wait before next attempt (workflow may not have started yet)
            time.sleep(3)
        except Exception:
            time.sleep(3)

    return None


def get_workflow_run_id(
    github_token: str,
    triggered_after: str,
    repo_owner: str = "jeleff1000",
    repo_name: str = "yahoo_oauth",
    max_attempts: int = 5,
    user_id: str = None,
) -> Optional[int]:
    """
    Find the workflow run ID that was triggered after a given time.

    DEPRECATED: Prefer get_workflow_run_id_from_motherduck() which uses user_id
    for exact matching instead of timestamp-based guessing.

    Args:
        github_token: GitHub personal access token
        triggered_after: ISO timestamp - find runs created after this time
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        max_attempts: Number of times to poll before giving up
        user_id: Optional user_id to try MotherDuck lookup first

    Returns:
        Run ID if found, None otherwise
    """
    import time

    # Try MotherDuck first if we have user_id (more reliable)
    if user_id:
        run_id = get_workflow_run_id_from_motherduck(user_id, max_attempts=max_attempts)
        if run_id:
            return run_id

    # Fallback to timestamp-based matching (less reliable with concurrent users)
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/league_import_worker.yml/runs"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }

    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, params={'per_page': 5}, timeout=30)
            if response.status_code == 200:
                runs = response.json().get('workflow_runs', [])
                for run in runs:
                    # Check if this run was created after our trigger time
                    if run['created_at'] >= triggered_after:
                        return run['id']

            # Wait before next attempt
            time.sleep(2)
        except Exception:
            time.sleep(2)

    return None


def get_workflow_jobs(
    run_id: int,
    github_token: str,
    repo_owner: str = "jeleff1000",
    repo_name: str = "yahoo_oauth",
) -> Dict:
    """
    Get the jobs and steps for a workflow run.

    Returns detailed progress information including:
    - Overall run status
    - Each job and its status
    - Each step within each job and its status
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}/jobs"
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            jobs = data.get('jobs', [])

            job_info = []
            total_steps = 0
            completed_steps = 0

            for job in jobs:
                steps = job.get('steps', [])
                job_steps = []

                for step in steps:
                    step_info = {
                        'name': step.get('name'),
                        'status': step.get('status'),  # queued, in_progress, completed
                        'conclusion': step.get('conclusion'),  # success, failure, skipped, etc.
                        'number': step.get('number'),
                    }
                    job_steps.append(step_info)
                    total_steps += 1
                    if step.get('status') == 'completed':
                        completed_steps += 1

                job_info.append({
                    'name': job.get('name'),
                    'status': job.get('status'),
                    'conclusion': job.get('conclusion'),
                    'started_at': job.get('started_at'),
                    'completed_at': job.get('completed_at'),
                    'steps': job_steps,
                })

            # Calculate progress percentage
            progress_pct = (completed_steps / total_steps * 100) if total_steps > 0 else 0

            return {
                'success': True,
                'run_id': run_id,
                'jobs': job_info,
                'total_steps': total_steps,
                'completed_steps': completed_steps,
                'progress_pct': progress_pct,
            }
        else:
            return {
                'success': False,
                'error': f"GitHub API error: {response.status_code}",
            }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Network error: {str(e)}"
        }


def check_import_status(
    user_id: str,
    github_token: str,
    repo_owner: str = "jeleff1000",
    repo_name: str = "yahoo_oauth",
    run_id: Optional[int] = None,
) -> Dict:
    """
    Check the status of an import job

    Args:
        user_id: The unique user ID from trigger_import_workflow
        github_token: GitHub personal access token
        repo_owner: GitHub repository owner
        repo_name: GitHub repository name
        run_id: Optional specific run ID to check (if known)

    Returns:
        Dictionary with status information including job steps
    """
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {github_token}'
    }

    try:
        # If we have a specific run_id, get that run directly
        if run_id:
            url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/runs/{run_id}"
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                run = response.json()
                # Get detailed job info
                jobs_info = get_workflow_jobs(run_id, github_token, repo_owner, repo_name)

                return {
                    'success': True,
                    'run_id': run_id,
                    'status': run['status'],  # queued, in_progress, completed
                    'conclusion': run.get('conclusion'),  # success, failure, cancelled, etc.
                    'run_url': run['html_url'],
                    'created_at': run['created_at'],
                    'updated_at': run['updated_at'],
                    'jobs': jobs_info.get('jobs', []),
                    'progress_pct': jobs_info.get('progress_pct', 0),
                    'total_steps': jobs_info.get('total_steps', 0),
                    'completed_steps': jobs_info.get('completed_steps', 0),
                }

        # Otherwise, get recent workflow runs and find the most recent
        url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/actions/workflows/league_import_worker.yml/runs"
        response = requests.get(url, headers=headers, params={'per_page': 5}, timeout=30)

        if response.status_code == 200:
            runs = response.json().get('workflow_runs', [])

            if runs:
                latest_run = runs[0]
                run_id = latest_run['id']

                # Get detailed job info
                jobs_info = get_workflow_jobs(run_id, github_token, repo_owner, repo_name)

                return {
                    'success': True,
                    'run_id': run_id,
                    'status': latest_run['status'],
                    'conclusion': latest_run.get('conclusion'),
                    'run_url': latest_run['html_url'],
                    'created_at': latest_run['created_at'],
                    'updated_at': latest_run['updated_at'],
                    'jobs': jobs_info.get('jobs', []),
                    'progress_pct': jobs_info.get('progress_pct', 0),
                    'total_steps': jobs_info.get('total_steps', 0),
                    'completed_steps': jobs_info.get('completed_steps', 0),
                }
            else:
                return {
                    'success': False,
                    'error': 'No workflow runs found'
                }
        else:
            return {
                'success': False,
                'error': f"GitHub API error: {response.status_code}",
                'details': response.text
            }

    except requests.exceptions.RequestException as e:
        return {
            'success': False,
            'error': f"Network error: {str(e)}"
        }


# Example usage for testing
if __name__ == "__main__":
    # This is an example - you would get these values from your Streamlit app
    example_league_data = {
        "league_id": "449.l.198278",
        "league_name": "Test League",
        "season": 2024,
        "start_year": 2020,
        "oauth_token": {
            "access_token": "your_access_token",
            "refresh_token": "your_refresh_token",
            "token_type": "bearer",
            "expires_in": 3600
        },
        "num_teams": 10,
        "playoff_teams": 6,
        "regular_season_weeks": 14
    }

    # Get GitHub token from environment
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("ERROR: Set GITHUB_TOKEN environment variable")
        exit(1)

    print("Triggering workflow...")
    result = trigger_import_workflow(example_league_data, github_token)

    print(json.dumps(result, indent=2))

    if result['success']:
        print(f"\n✅ Import started!")
        print(f"User ID: {result['user_id']}")
        print(f"Track progress: {result['workflow_run_url']}")
