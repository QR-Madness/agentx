import React, { useState, useCallback } from 'react';
import { 
  Play, 
  Square, 
  Pause, 
  Download,
  Brain,
  Wrench,
  Sparkles,
  ChevronRight,
  ChevronDown,
  Clock,
  Zap,
  CheckCircle2,
  XCircle,
  Loader2
} from 'lucide-react';
import { api, AgentRunRequest, ReasoningStep, ToolUsage } from '../../lib/api';
import '../../styles/AgentTab.css';

type AgentStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

interface AgentRun {
  id: string;
  task: string;
  status: AgentStatus;
  startedAt: Date;
  completedAt?: Date;
  result?: string;
  reasoningTrace: ReasoningStep[];
  toolsUsed: ToolUsage[];
  tokensUsed?: number;
  error?: string;
}

export const AgentTab: React.FC = () => {
  const [task, setTask] = useState('');
  const [currentRun, setCurrentRun] = useState<AgentRun | null>(null);
  const [expandedTrace, setExpandedTrace] = useState(true);
  const [expandedTools, setExpandedTools] = useState(true);

  const handleRun = useCallback(async () => {
    if (!task.trim()) return;

    const newRun: AgentRun = {
      id: Date.now().toString(),
      task,
      status: 'running',
      startedAt: new Date(),
      reasoningTrace: [],
      toolsUsed: [],
    };

    setCurrentRun(newRun);

    try {
      const request: AgentRunRequest = {
        task,
        reasoning_strategy: 'auto',
      };

      const response = await api.runAgent(request);

      setCurrentRun(prev => prev ? {
        ...prev,
        status: 'completed',
        completedAt: new Date(),
        result: response.result,
        reasoningTrace: response.reasoning_trace || [],
        toolsUsed: response.tools_used || [],
        tokensUsed: response.tokens_used,
      } : null);
    } catch (err) {
      setCurrentRun(prev => prev ? {
        ...prev,
        status: 'error',
        completedAt: new Date(),
        error: err instanceof Error ? err.message : 'An error occurred',
      } : null);
    }
  }, [task]);

  const handleCancel = useCallback(() => {
    setCurrentRun(prev => prev ? { ...prev, status: 'idle' } : null);
  }, []);

  const getStatusIcon = (status: AgentStatus) => {
    switch (status) {
      case 'running':
        return <Loader2 className="spin" size={16} />;
      case 'completed':
        return <CheckCircle2 size={16} />;
      case 'error':
        return <XCircle size={16} />;
      default:
        return null;
    }
  };

  const getStepIcon = (type: ReasoningStep['type']) => {
    switch (type) {
      case 'thought':
        return <Brain size={14} />;
      case 'action':
        return <Zap size={14} />;
      case 'observation':
        return <Sparkles size={14} />;
      case 'reflection':
        return <Clock size={14} />;
      default:
        return null;
    }
  };

  return (
    <div className="agent-tab">
      {/* Header */}
      <div className="agent-header fade-in">
        <div className="header-content">
          <h1 className="page-title">
            <Brain className="page-icon-svg" />
            <span>Agent (Unstable & Under Heavy Construction)</span>
          </h1>
          <p className="page-subtitle">Execute tasks with AI-powered reasoning</p>
        </div>
      </div>

      {/* Task Input */}
      <div className="task-input-section card">
        <div className="task-input-wrapper">
          <textarea
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Describe your task... (e.g., 'Analyze this codebase and suggest improvements')"
            className="task-input"
            rows={3}
          />
          <div className="task-actions">
            {currentRun?.status === 'running' ? (
              <>
                <button className="button-secondary" onClick={handleCancel}>
                  <Square size={16} />
                  Cancel
                </button>
                <button className="button-secondary" disabled>
                  <Pause size={16} />
                  Pause
                </button>
              </>
            ) : (
              <button 
                className="button-primary run-button" 
                onClick={handleRun}
                disabled={!task.trim()}
              >
                <Play size={16} />
                Run Agent
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="agent-content">
        {/* Reasoning Trace Panel */}
        <div className="trace-panel card">
          <div 
            className="panel-header"
            onClick={() => setExpandedTrace(!expandedTrace)}
          >
            <div className="panel-title">
              {expandedTrace ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
              <Brain size={18} className="panel-icon" />
              <span>Reasoning Trace</span>
            </div>
            {currentRun && (
              <div className={`status-badge ${currentRun.status}`}>
                {getStatusIcon(currentRun.status)}
                <span>{currentRun.status}</span>
              </div>
            )}
          </div>
          
          {expandedTrace && (
            <div className="panel-content">
              {!currentRun ? (
                <div className="empty-state">
                  <Sparkles size={32} className="empty-icon" />
                  <p>Run a task to see the reasoning trace</p>
                </div>
              ) : currentRun.reasoningTrace.length === 0 && currentRun.status === 'running' ? (
                <div className="loading-state">
                  <Loader2 size={24} className="spin" />
                  <p>Agent is thinking...</p>
                </div>
              ) : currentRun.reasoningTrace.length === 0 ? (
                <div className="empty-state">
                  <p>No reasoning steps recorded</p>
                </div>
              ) : (
                <div className="trace-list">
                  {currentRun.reasoningTrace.map((step, index) => (
                    <div key={index} className={`trace-step ${step.type}`}>
                      <div className="step-icon">{getStepIcon(step.type)}</div>
                      <div className="step-content">
                        <div className="step-type">{step.type}</div>
                        <div className="step-text">{step.content}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Tools Panel */}
        <div className="tools-panel card">
          <div 
            className="panel-header"
            onClick={() => setExpandedTools(!expandedTools)}
          >
            <div className="panel-title">
              {expandedTools ? <ChevronDown size={18} /> : <ChevronRight size={18} />}
              <Wrench size={18} className="panel-icon" />
              <span>Tool Usage</span>
            </div>
            {currentRun && currentRun.toolsUsed.length > 0 && (
              <span className="tool-count">{currentRun.toolsUsed.length} tools</span>
            )}
          </div>
          
          {expandedTools && (
            <div className="panel-content">
              {!currentRun || currentRun.toolsUsed.length === 0 ? (
                <div className="empty-state">
                  <Wrench size={32} className="empty-icon" />
                  <p>No tools used yet</p>
                </div>
              ) : (
                <div className="tools-list">
                  {currentRun.toolsUsed.map((tool, index) => (
                    <div key={index} className={`tool-item ${tool.success ? 'success' : 'error'}`}>
                      <div className="tool-header">
                        {tool.success ? (
                          <CheckCircle2 size={14} className="tool-status-icon success" />
                        ) : (
                          <XCircle size={14} className="tool-status-icon error" />
                        )}
                        <span className="tool-name">{tool.tool}</span>
                        {tool.duration_ms && (
                          <span className="tool-duration">{tool.duration_ms}ms</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Result Section */}
      {currentRun?.result && (
        <div className="result-section card fade-in">
          <div className="result-header">
            <h3 className="section-title">
              <Sparkles size={18} className="section-title-icon" />
              Result
            </h3>
            <button className="button-ghost">
              <Download size={16} />
              Export
            </button>
          </div>
          <div className="result-content">
            <p>{currentRun.result}</p>
          </div>
          {currentRun.tokensUsed && (
            <div className="result-meta">
              <span className="meta-item">
                <Zap size={14} />
                {currentRun.tokensUsed.toLocaleString()} tokens
              </span>
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {currentRun?.error && (
        <div className="error-section card fade-in">
          <div className="error-content">
            <XCircle size={20} />
            <p>{currentRun.error}</p>
          </div>
        </div>
      )}
    </div>
  );
};
