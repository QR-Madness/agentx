import React, { useState, useRef } from 'react';
import styled from 'styled-components';

const ViewContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  padding: 24px;
`;

const ViewHeader = styled.div`
  margin-bottom: 24px;

  h2 {
    font-size: 24px;
    font-weight: 600;
    color: ${({ theme }) => theme.colors.textPrimary};
  }
`;

const FileAnalysisContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  height: 100%;
`;

const DropZone = styled.div<{ $isDragging: boolean }>`
  border: 2px dashed ${({ theme, $isDragging }) =>
    $isDragging ? theme.colors.accentPrimary : theme.colors.borderColor};
  border-radius: 12px;
  padding: 60px 40px;
  text-align: center;
  cursor: pointer;
  transition: all 0.2s;
  background: ${({ theme, $isDragging }) =>
    $isDragging ? theme.colors.bgTertiary : theme.colors.bgSecondary};

  &:hover {
    border-color: ${({ theme }) => theme.colors.accentPrimary};
    background: ${({ theme }) => theme.colors.bgTertiary};
  }
`;

const DropIcon = styled.span`
  font-size: 48px;
  display: block;
  margin-bottom: 16px;
`;

const DropText = styled.p`
  color: ${({ theme }) => theme.colors.textSecondary};
  font-size: 16px;
`;

const FileList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 200px;
  overflow-y: auto;
`;

const FileItem = styled.div`
  padding: 12px;
  background: ${({ theme }) => theme.colors.bgSecondary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 8px;
  color: ${({ theme }) => theme.colors.textPrimary};
`;

const AnalysisResults = styled.div`
  flex: 1;
  background: ${({ theme }) => theme.colors.bgSecondary};
  border: 1px solid ${({ theme }) => theme.colors.borderColor};
  border-radius: 12px;
  padding: 20px;
  overflow-y: auto;

  h3 {
    color: ${({ theme }) => theme.colors.textPrimary};
    margin-bottom: 16px;
  }

  p {
    color: ${({ theme }) => theme.colors.textSecondary};
    margin-bottom: 8px;
  }
`;

const PlaceholderText = styled.p`
  color: ${({ theme }) => theme.colors.textMuted};
  text-align: center;
  padding: 40px;
  font-style: italic;
`;

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i];
};

export const FileAnalysisView: React.FC = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    const newFiles = Array.from(fileList);
    setFiles(newFiles);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    handleFiles(e.target.files);
  };

  return (
    <ViewContainer>
      <ViewHeader>
        <h2>File Analysis</h2>
      </ViewHeader>
      <FileAnalysisContainer>
        <DropZone
          $isDragging={isDragging}
          onClick={handleClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <DropIcon>📄</DropIcon>
          <DropText>Drop files here or click to browse</DropText>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            onChange={handleFileInput}
          />
        </DropZone>

        {files.length > 0 && (
          <FileList>
            {files.map((file, idx) => (
              <FileItem key={idx}>
                📄 {file.name} ({formatFileSize(file.size)})
              </FileItem>
            ))}
          </FileList>
        )}

        <AnalysisResults>
          {files.length === 0 ? (
            <PlaceholderText>Upload files to analyze them</PlaceholderText>
          ) : (
            <>
              <h3>Analysis Results</h3>
              <p>Files processed: {files.length}</p>
              <PlaceholderText>
                Detailed analysis functionality will be implemented here. This will
                include file content analysis, metadata extraction, and AI-powered
                insights.
              </PlaceholderText>
            </>
          )}
        </AnalysisResults>
      </FileAnalysisContainer>
    </ViewContainer>
  );
};
