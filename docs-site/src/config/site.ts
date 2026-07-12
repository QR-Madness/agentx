// Site-wide metadata. Single source of truth for titles, links, and copyright.
export const site = {
  name: 'AgentX',
  title: 'AgentX',
  description:
    'A self-hosted, glassbox Cognitive OS: persistent memory, Agent Teams that collaborate, reasoning, and MCP tools — on any model you bring, behind a REST API you run yourself.',
  // Where the documentation section lives within the hub.
  docsBasePath: '/docs',
  repoUrl: 'https://github.com/QR-Madness/agentx',
  repoName: 'agentx',
  copyright: `Copyright © ${new Date().getFullYear()} AgentX`,
} as const;
