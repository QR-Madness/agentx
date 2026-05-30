// Site-wide metadata. Single source of truth for titles, links, and copyright.
export const site = {
  name: 'AgentX',
  title: 'AgentX',
  description:
    'A self-hosted, glassbox AI agent platform: persistent memory, multi-agent Agent Alloy orchestration, MCP tools, and bring-your-own models — behind a REST API you run yourself.',
  // Where the documentation section lives within the hub.
  docsBasePath: '/docs',
  repoUrl: 'https://github.com/QR-Madness/agentx',
  repoName: 'agentx',
  copyright: `Copyright © ${new Date().getFullYear()} AgentX`,
} as const;
