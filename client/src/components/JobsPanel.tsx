/**
 * JobsPanel - Monitoring panel for consolidation jobs
 * Shows job status, metrics, history, and allows manual triggering.
 */

import React, { useState } from 'react';
import {
  Clock,
  Play,
  Pause,
  RefreshCw,
  ChevronDown,
  ChevronRight,
  CheckCircle,
  XCircle,
  AlertCircle,
  Activity
} from 'lucide-react';
import { useJobs, useJob } from '../lib/hooks';
import { api, JobStatus } from '../lib/api';

// Format duration for display
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// Format timestamp for display
function formatTimestamp(timestamp: string | null): string {
  if (!timestamp) return 'Never';
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  } catch {
    return 'Unknown';
  }
}

// Single job row component
function JobRow({ job, onRefresh }: { job: JobStatus; onRefresh: () => void }) {
  const [expanded, setExpanded] = useState(false);
  const [running, setRunning] = useState(false);
  const [toggling, setToggling] = useState(false);
  const { history, refresh: refreshHistory } = useJob(expanded ? job.name : '');

  const handleRun = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setRunning(true);
    try {
      await api.runJob(job.name);
      onRefresh();
      if (expanded) refreshHistory();
    } catch (err) {
      console.error('Failed to run job:', err);
    } finally {
      setRunning(false);
    }
  };

  const handleToggle = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setToggling(true);
    try {
      const enabled = job.status === 'disabled';
      await api.toggleJob(job.name, enabled);
      onRefresh();
    } catch (err) {
      console.error('Failed to toggle job:', err);
    } finally {
      setToggling(false);
    }
  };

  const handleExpand = () => {
    setExpanded(!expanded);
    if (!expanded) refreshHistory();
  };

  const statusIcon = () => {
    switch (job.status) {
      case 'running':
        return <RefreshCw size={14} className="spin job-status-icon running" />;
      case 'disabled':
        return <Pause size={14} className="job-status-icon disabled" />;
      default:
        return <Activity size={14} className="job-status-icon idle" />;
    }
  };

  return (
    <div className={`job-row ${expanded ? 'expanded' : ''}`}>
      <div className="job-summary" onClick={handleExpand}>
        <span className="job-expand-icon">
          {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </span>
        <span className="job-name">{job.name}</span>
        <span className={`job-status-badge ${job.status}`}>
          {statusIcon()}
          <span>{job.status}</span>
        </span>
        <span className="job-last-run">{formatTimestamp(job.last_run)}</span>
        <span className="job-success-rate">
          {(job.success_rate * 100).toFixed(0)}%
        </span>
        <div className="job-actions">
          <button
            className="button-ghost job-action-btn"
            onClick={handleRun}
            disabled={running || job.status === 'running'}
            title="Run now"
          >
            {running ? <RefreshCw size={14} className="spin" /> : <Play size={14} />}
          </button>
          <button
            className="button-ghost job-action-btn"
            onClick={handleToggle}
            disabled={toggling}
            title={job.status === 'disabled' ? 'Enable' : 'Disable'}
          >
            {toggling ? (
              <RefreshCw size={14} className="spin" />
            ) : job.status === 'disabled' ? (
              <Play size={14} />
            ) : (
              <Pause size={14} />
            )}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="job-details">
          <p className="job-description">{job.description}</p>

          <div className="job-metrics-grid">
            <div className="job-metric">
              <span className="metric-value">{job.run_count}</span>
              <span className="metric-label">Total Runs</span>
            </div>
            <div className="job-metric">
              <span className="metric-value">{job.success_count}</span>
              <span className="metric-label">Successes</span>
            </div>
            <div className="job-metric">
              <span className="metric-value">{job.failure_count}</span>
              <span className="metric-label">Failures</span>
            </div>
            <div className="job-metric">
              <span className="metric-value">{formatDuration(job.avg_duration_ms)}</span>
              <span className="metric-label">Avg Duration</span>
            </div>
          </div>

          <div className="job-info-row">
            <span className="info-label">Interval:</span>
            <span className="info-value">{job.interval_minutes} minutes</span>
          </div>

          {job.last_error && (
            <div className="job-error">
              <AlertCircle size={14} />
              <span>Last error: {job.last_error}</span>
            </div>
          )}

          {history.length > 0 && (
            <div className="job-history">
              <h4>Recent Runs</h4>
              <div className="history-list">
                {history.slice(0, 5).map((entry, idx) => (
                  <div key={idx} className={`history-entry ${entry.success ? 'success' : 'failure'}`}>
                    {entry.success ? (
                      <CheckCircle size={14} className="history-icon success" />
                    ) : (
                      <XCircle size={14} className="history-icon failure" />
                    )}
                    <span className="history-time">{formatTimestamp(entry.timestamp)}</span>
                    <span className="history-duration">{formatDuration(entry.duration_ms)}</span>
                    <span className="history-items">{entry.items_processed} items</span>
                    {entry.error && <span className="history-error">{entry.error}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Main JobsPanel component
export const JobsPanel: React.FC = () => {
  const { jobs, worker, loading, error, refresh } = useJobs();

  if (loading) {
    return (
      <div className="jobs-panel-loading">
        <RefreshCw size={24} className="spin" />
        <p>Loading jobs...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="jobs-panel-error">
        <AlertCircle size={24} />
        <p>Failed to load jobs: {error.message}</p>
        <button className="button-secondary" onClick={refresh}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="jobs-panel">
      <div className="jobs-header">
        <div className="jobs-title">
          <Clock size={18} />
          <span>Background Jobs</span>
        </div>
        <button className="button-ghost" onClick={refresh} title="Refresh">
          <RefreshCw size={16} />
        </button>
      </div>

      {worker && (
        <div className="worker-status">
          <span className={`worker-indicator ${worker.status}`}></span>
          <span>Worker: {worker.id}</span>
          <span className="worker-uptime">
            Uptime: {Math.floor(worker.uptime_seconds / 60)}m
          </span>
          <span className="worker-jobs-run">
            Jobs run: {worker.jobs_run}
          </span>
        </div>
      )}

      <div className="jobs-list-header">
        <span className="col-expand"></span>
        <span className="col-name">Job</span>
        <span className="col-status">Status</span>
        <span className="col-last-run">Last Run</span>
        <span className="col-success">Success</span>
        <span className="col-actions">Actions</span>
      </div>

      <div className="jobs-list">
        {jobs.map(job => (
          <JobRow key={job.name} job={job} onRefresh={refresh} />
        ))}
      </div>

      {jobs.length === 0 && (
        <div className="jobs-empty">
          <Clock size={32} />
          <p>No jobs configured</p>
        </div>
      )}
    </div>
  );
};

export default JobsPanel;
