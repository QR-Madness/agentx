// Site-wide metadata. Single source of truth for titles, links, and copyright.
export const site = {
  name: 'AgentX',
  title: 'AgentX',
  description:
    'AI Agent Platform with MCP client integration, multi-model reasoning, drafting strategies, and a persistent memory system.',
  // Where the documentation section lives within the hub.
  docsBasePath: '/docs',
  repoUrl: 'https://github.com/QR-Madness/agentx',
  repoName: 'agentx',
  copyright: `Copyright © ${new Date().getFullYear()} AgentX`,
} as const;
