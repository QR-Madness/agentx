/**
 * VersionMismatchPage - Shown when client/server versions are incompatible
 */

import { AlertTriangle, RefreshCw } from 'lucide-react';
import type { VersionInfo } from '../lib/api';
import { CLIENT_VERSION, CLIENT_PROTOCOL_VERSION, compareSemver } from '../lib/api';
import './VersionMismatchPage.css';

interface VersionMismatchPageProps {
  versionInfo: VersionInfo | null;
  onRetry: () => void;
}

export function VersionMismatchPage({ versionInfo, onRetry }: VersionMismatchPageProps) {
  const isProtocolMismatch =
    versionInfo && versionInfo.protocol_version !== CLIENT_PROTOCOL_VERSION;

  const isClientOutdated =
    versionInfo &&
    !isProtocolMismatch &&
    compareSemver(CLIENT_VERSION, versionInfo.min_client_version) < 0;

  return (
    <div className="version-mismatch-page">
      <div className="version-mismatch-container">
        <div className="version-mismatch-icon">
          <AlertTriangle size={48} />
        </div>

        <h1 className="version-mismatch-title">Version Mismatch</h1>

        <p className="version-mismatch-message">
          {isProtocolMismatch
            ? 'The client and server use incompatible protocols.'
            : isClientOutdated
              ? 'Your client version is too old to connect to this server.'
              : 'Version compatibility check failed.'}
        </p>

        <div className="version-details">
          <div className="version-row">
            <span className="version-label">Client Version</span>
            <span className="version-value">{CLIENT_VERSION}</span>
          </div>
          <div className="version-row">
            <span className="version-label">Client Protocol</span>
            <span className="version-value">{CLIENT_PROTOCOL_VERSION}</span>
          </div>
          {versionInfo && (
            <>
              <div className="version-divider" />
              <div className="version-row">
                <span className="version-label">Server Version</span>
                <span className="version-value">{versionInfo.version}</span>
              </div>
              <div className="version-row">
                <span className="version-label">Server Protocol</span>
                <span className="version-value">{versionInfo.protocol_version}</span>
              </div>
              <div className="version-row">
                <span className="version-label">Minimum Client</span>
                <span className="version-value">{versionInfo.min_client_version}</span>
              </div>
            </>
          )}
        </div>

        <div className="version-mismatch-actions">
          <button onClick={onRetry} className="button-primary version-retry-btn">
            <RefreshCw size={18} />
            <span>Retry Connection</span>
          </button>
        </div>

        <p className="version-mismatch-hint">
          {isProtocolMismatch
            ? 'Both client and server need to be updated to compatible versions.'
            : isClientOutdated
              ? 'Update your client to continue using this server.'
              : 'Check your server configuration and try again.'}
        </p>
      </div>
    </div>
  );
}
